const path = require("path");
const { spawn } = require("child_process");

function runInference({ inputPath, outputsDir, checkpointPath }) {
  const pythonBin = process.env.PYTHON_BIN || "python3";
  const scriptPath = path.join(__dirname, "..", "python", "inference.py");
  const outputFilename = `comic-${Date.now()}-${Math.round(Math.random() * 1e6)}.png`;
  const outputPath = path.join(outputsDir, outputFilename);

  return new Promise((resolve, reject) => {
    const args = [
      scriptPath,
      "--input",
      inputPath,
      "--output",
      outputPath,
      "--checkpoint",
      checkpointPath,
    ];

    const proc = spawn(pythonBin, args, { stdio: ["ignore", "pipe", "pipe"] });
    let stderr = "";

    proc.stderr.on("data", (data) => {
      stderr += data.toString();
    });

    proc.on("error", (err) => {
      reject(err);
    });

    proc.on("close", (code) => {
      if (code === 0) {
        resolve({ outputPath, outputFilename });
        return;
      }
      const error = new Error(`Inference failed with code ${code}. ${stderr}`);
      error.details = stderr;
      reject(error);
    });
  });
}

module.exports = {
  runInference,
};
