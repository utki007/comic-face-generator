const path = require("path");
const express = require("express");
const morgan = require("morgan");

const indexRoutes = require("./routes/index");
const generateRoutes = require("./routes/generate");

const app = express();
const PORT = process.env.PORT || 3000;

app.set("view engine", "ejs");
app.set("views", path.join(__dirname, "views"));

app.use(morgan("dev"));
app.use(express.json());
app.use(express.urlencoded({ extended: true }));

const publicDir = path.join(__dirname, "public");
const uploadsDir = path.join(__dirname, "uploads");
const outputsDir = path.join(__dirname, "outputs");

app.use("/static", express.static(publicDir));
app.use("/uploads", express.static(uploadsDir));
app.use("/outputs", express.static(outputsDir));

app.use("/", indexRoutes);
app.use("/generate", generateRoutes);

app.use((err, req, res, next) => {
  const status = err.status || 500;
  const message = err.message || "Server error";
  res.status(status).json({ success: false, error: message });
});

app.listen(PORT, () => {
  console.log(`Face2Comic server running on http://localhost:${PORT}`);
});
