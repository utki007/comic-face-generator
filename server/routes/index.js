const express = require("express");

const router = express.Router();

router.get("/", (req, res) => {
  res.render("index", { title: "Face2Comic" });
});

module.exports = router;
