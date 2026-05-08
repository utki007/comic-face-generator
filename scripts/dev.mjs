import { spawn } from 'child_process';
import { existsSync } from 'fs';
import { dirname, resolve } from 'path';
import { fileURLToPath } from 'url';

const scriptDir = dirname(fileURLToPath(import.meta.url));
const repoRoot = resolve(scriptDir, '..');
const frontendDir = resolve(repoRoot, 'frontend');

const pythonCandidates = [
  resolve(repoRoot, '.venv', 'bin', 'python'),
  resolve(repoRoot, '.venv', 'Scripts', 'python.exe'),
  'python3',
  'python',
];

const python = pythonCandidates.find((candidate) => {
  return candidate === 'python3' || candidate === 'python' || existsSync(candidate);
}) ?? 'python3';

function startProcess(command, args, cwd, label) {
  const child = spawn(command, args, {
    cwd,
    stdio: 'inherit',
    shell: false,
  });

  child.on('exit', (code, signal) => {
    if (signal) {
      console.log(`${label} exited with signal ${signal}`);
    } else {
      console.log(`${label} exited with code ${code ?? 0}`);
    }

    process.exitCode = code ?? 0;
    shutdown();
  });

  child.on('error', (error) => {
    console.error(`Failed to start ${label}:`, error.message);
    process.exitCode = 1;
    shutdown();
  });

  return child;
}

const api = startProcess(
  python,
  ['-m', 'uvicorn', 'api:app', '--reload', '--port', '8000'],
  repoRoot,
  'API',
);

const ui = startProcess(
  'npm',
  ['run', 'dev', '--', '--host', '127.0.0.1'],
  frontendDir,
  'UI',
);

function shutdown() {
  for (const child of [api, ui]) {
    if (child && !child.killed) {
      child.kill('SIGTERM');
    }
  }
}

for (const signal of ['SIGINT', 'SIGTERM']) {
  process.on(signal, () => {
    process.exitCode = 0;
    shutdown();
  });
}