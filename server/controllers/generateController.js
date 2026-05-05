const path = require("path");

const inferenceService = require("../services/inferenceService");

async function generateComic(req, res, next) {
  try {
    if (!req.file) {
      return res.status(400).json({ success: false, error: "No image uploaded." });
    }

    const inputPath = req.file.path;
    const outputsDir = path.join(__dirname, "..", "outputs");
    const checkpointPath =
      process.env.MODEL_CHECKPOINT ||
      path.join(__dirname, "..", "..", "checkpoints", "pix2pix_best_epoch_066.pt");

    const { outputFilename } = await inferenceService.runInference({
      inputPath,
      outputsDir,
      checkpointPath,
    });

    return res.json({
      success: true,
      outputUrl: `/outputs/${outputFilename}`,
    });
  } catch (err) {
    return next(err);
  }
}

module.exports = {
  generateComic,
};
