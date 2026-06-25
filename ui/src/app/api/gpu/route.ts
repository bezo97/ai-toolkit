import { NextResponse } from 'next/server';
import { exec, execSync } from 'child_process';
import { promisify } from 'util';
import { createRequire } from 'module';
import os from 'os';
import path from 'path';
import fs from 'fs';

const execAsync = promisify(exec);

interface MacGpuResult {
  name: string;
  memUsed: number;
  memTotal: number;
  gpuLoad: number;
  temperature: number;
  fanSpeed: number;
  powerDraw: number;
}

async function getMacGpuInfo(): Promise<MacGpuResult | null> {
  try {
    const memoryTotal = os.totalmem() / (1024 * 1024);

    // Get GPU name and core count from system_profiler
    let gpuName = 'Apple GPU';
    try {
      const spOut = execSync(
        'system_profiler SPDisplaysDataType 2>/dev/null | grep -E "Chipset Model|Total Number of Cores"',
        { encoding: 'utf-8', timeout: 5000 },
      );
      const nameMatch = spOut.match(/Chipset Model:\s*(.+)/);
      const coresMatch = spOut.match(/Total Number of Cores:\s*(\d+)/);
      if (nameMatch) {
        gpuName = nameMatch[1].trim();
        if (coresMatch) {
          gpuName += ` GPU (${coresMatch[1]} cores)`;
        }
      }
    } catch {
      // fallback to generic name
    }

    let temperature = 0;
    let gpuLoad = 0;
    let fanSpeed = 0;
    let powerDraw = 0;
    let memUsed = 0;
    let memTotal = memoryTotal;

    try {
      // Use createRequire to hide from webpack static analysis so it doesn't fail on non-mac platforms
      const nativeRequire = createRequire(import.meta.url);
      const ms = nativeRequire('macstats') as any;

      try {
        const gpuData = ms.getGpuDataSync();
        temperature = gpuData.temperature || 0;
        gpuLoad = gpuData.usage || 0;
      } catch {
        // ignore
      }

      try {
        const fanData = ms.getFanDataSync();
        const fanKeys = Object.keys(fanData);
        if (fanKeys.length > 0) {
          fanSpeed = fanData[fanKeys[0]].rpm || 0;
        }
      } catch {
        // ignore
      }

      try {
        const powerData = ms.getPowerDataSync();
        powerDraw = powerData.gpu || 0;
      } catch {
        // ignore
      }

      try {
        const ramData = ms.getRAMUsageSync();
        memUsed = ramData.used / (1024 * 1024);
        memTotal = ramData.total / (1024 * 1024);
      } catch {
        // ignore
      }
    } catch (error) {
      console.warn('macstats not available:', error);
    }

    return { name: gpuName, memUsed, memTotal, gpuLoad, temperature, fanSpeed, powerDraw };
  } catch {
    return null;
  }
}

export async function GET() {
  try {
    // Get platform
    const platform = os.platform();
    const isWindows = platform === 'win32';
    const isMac = platform === 'darwin';

    if (isMac) {
      const macGpu = await getMacGpuInfo();
      if (macGpu) {
        return NextResponse.json({
          hasNvidiaSmi: false,
          isMac: true,
          gpus: [
            {
              index: 0,
              name: macGpu.name,
              driverVersion: 'macOS',
              temperature: Math.round(macGpu.temperature),
              utilization: {
                gpu: macGpu.gpuLoad,
                memory: macGpu.memTotal > 0 ? Math.round((macGpu.memUsed / macGpu.memTotal) * 100) : 0,
              },
              memory: {
                total: Math.round(macGpu.memTotal),
                free: Math.round(macGpu.memTotal - macGpu.memUsed),
                used: Math.round(macGpu.memUsed),
              },
              power: { draw: macGpu.powerDraw, limit: 0 },
              clocks: { graphics: 0, memory: 0 },
              fan: { speed: macGpu.fanSpeed },
            },
          ],
        });
      }
      return NextResponse.json({
        hasNvidiaSmi: false,
        isMac: true,
        gpus: [],
        error: 'Could not read Mac GPU stats',
      });
    }

    // Check if nvidia-smi is available
    const hasNvidiaSmi = await checkNvidiaSmi(isWindows);

    if (!hasNvidiaSmi) {
      // Fallback: use PyTorch to detect GPUs (works on both NVIDIA and ROCm)
      const pytorchGpus = await getPytorchGpus();
      if (pytorchGpus.length > 0) {
        return NextResponse.json({
          hasNvidiaSmi: false,
          isMac: false,
          gpus: pytorchGpus,
        });
      }
      return NextResponse.json({
        hasNvidiaSmi: false,
        isMac: false,
        gpus: [],
        error: 'nvidia-smi not found and no PyTorch CUDA devices detected',
      });
    }

    // Get GPU stats
    const gpuStats = await getGpuStats(isWindows);

    return NextResponse.json({
      hasNvidiaSmi: true,
      gpus: gpuStats,
    });
  } catch (error) {
    console.error('Error fetching NVIDIA GPU stats:', error);
    return NextResponse.json(
      {
        hasNvidiaSmi: false,
        isMac: false,
        gpus: [],
        error: `Failed to fetch GPU stats: ${error instanceof Error ? error.message : String(error)}`,
      },
      { status: 500 },
    );
  }
}

/**
 * Resolve the Python interpreter to use for GPU detection.
 * Tries the project venv first, then falls back to `python` / `python3`.
 */
function findPython(): string {
  // Walk up from this file (ui/src/app/api/gpu/route.ts) to project root
  const projectRoot = path.resolve(__dirname, '../../../..');
  const candidates = [
    path.join(projectRoot, 'venv', 'Scripts', 'python.exe'),   // Windows
    path.join(projectRoot, 'venv', 'bin', 'python'),            // Linux/macOS
    path.join(projectRoot, '.venv', 'Scripts', 'python.exe'),   // Windows alt
    path.join(projectRoot, '.venv', 'bin', 'python'),           // Linux/macOS alt
  ];
  for (const c of candidates) {
    if (fs.existsSync(c)) return c;
  }
  // Fallback: hope `python` is on PATH
  return 'python';
}

/**
 * Fallback GPU detection via PyTorch subprocess.
 * Works on both NVIDIA CUDA and AMD ROCm (torch.cuda.* shim).
 * Returns basic GPU info without live stats (no temp, utilization, etc.).
 */
async function getPytorchGpus(): Promise<any[]> {
  const python = findPython();
  const script = `
import torch, json
gpus = []
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        total_mem = 0
        try:
            total_mem = torch.cuda.mem_get_info(i)[1] // (1024**2)
        except Exception:
            total_mem = torch.cuda.get_device_properties(i).total_mem // (1024**2)
        gpus.append({"index": i, "name": torch.cuda.get_device_name(i), "total_mem": total_mem})
print(json.dumps(gpus))
`;
  // Write to temp file to avoid all shell quoting issues
  const tmpFile = path.join(os.tmpdir(), `aitk_gpu_detect_${Date.now()}.py`);
  try {
    fs.writeFileSync(tmpFile, script, 'utf-8');
    const { stdout } = await execAsync(`${python} ${tmpFile}`, {
      timeout: 20000,
    });
    const gpus: any[] = JSON.parse(stdout.trim());
    // Map to the GpuInfo shape with zeroed-out stats
    return gpus.map((g: any) => ({
      index: g.index,
      name: g.name,
      driverVersion: 'PyTorch',
      temperature: 0,
      utilization: { gpu: 0, memory: 0 },
      memory: { total: g.total_mem, free: g.total_mem, used: 0 },
      power: { draw: 0, limit: 0 },
      clocks: { graphics: 0, memory: 0 },
      fan: { speed: 0 },
    }));
  } catch (err) {
    console.warn('PyTorch GPU detection failed:', err instanceof Error ? err.message : String(err));
    return [];
  } finally {
    try { fs.unlinkSync(tmpFile); } catch { /* ignore cleanup errors */ }
  }
}

async function checkNvidiaSmi(isWindows: boolean): Promise<boolean> {
  try {
    if (isWindows) {
      // Check if nvidia-smi is available on Windows
      // It's typically located in C:\Program Files\NVIDIA Corporation\NVSMI\nvidia-smi.exe
      // but we'll just try to run it directly as it may be in PATH
      await execAsync('nvidia-smi -L');
    } else {
      // Linux/macOS check
      await execAsync('which nvidia-smi');
    }
    return true;
  } catch (error) {
    return false;
  }
}

async function getGpuStats(isWindows: boolean) {
  // Command is the same for both platforms, but the path might be different
  const command =
    'nvidia-smi --query-gpu=index,name,driver_version,temperature.gpu,utilization.gpu,utilization.memory,memory.total,memory.free,memory.used,power.draw,power.limit,clocks.current.graphics,clocks.current.memory,fan.speed --format=csv,noheader,nounits';

  // Execute command
  const { stdout } = await execAsync(command, {
    env: { ...process.env, CUDA_DEVICE_ORDER: 'PCI_BUS_ID' },
  });

  // Parse CSV output
  const gpus = stdout
    .trim()
    .split('\n')
    .map(line => {
      const [
        index,
        name,
        driverVersion,
        temperature,
        gpuUtil,
        memoryUtil,
        memoryTotal,
        memoryFree,
        memoryUsed,
        powerDraw,
        powerLimit,
        clockGraphics,
        clockMemory,
        fanSpeed,
      ] = line.split(', ').map(item => item.trim());

      return {
        index: parseInt(index),
        name,
        driverVersion,
        temperature: parseInt(temperature),
        utilization: {
          gpu: parseInt(gpuUtil),
          memory: parseInt(memoryUtil),
        },
        memory: {
          total: parseInt(memoryTotal),
          free: parseInt(memoryFree),
          used: parseInt(memoryUsed),
        },
        power: {
          draw: parseFloat(powerDraw),
          limit: parseFloat(powerLimit),
        },
        clocks: {
          graphics: parseInt(clockGraphics),
          memory: parseInt(clockMemory),
        },
        fan: {
          speed: parseInt(fanSpeed) || 0, // Some GPUs might not report fan speed, default to 0
        },
      };
    });

  return gpus;
}

