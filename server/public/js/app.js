const { useEffect, useMemo, useState } = React;

const e = React.createElement;

const TARGET_SIZE = 256;

function loadImageFromFile(file) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    const url = URL.createObjectURL(file);

    img.onload = () => {
      URL.revokeObjectURL(url);
      resolve(img);
    };

    img.onerror = () => {
      URL.revokeObjectURL(url);
      reject(new Error("Failed to load image."));
    };

    img.src = url;
  });
}

function loadImageFromUrl(url) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = () => reject(new Error("Failed to load image."));
    img.src = url;
  });
}

async function renderSquarePreview(img, size = TARGET_SIZE) {
  const canvas = document.createElement("canvas");
  canvas.width = size;
  canvas.height = size;
  const ctx = canvas.getContext("2d");

  if (!ctx) {
    throw new Error("Canvas not supported.");
  }

  const scale = Math.max(size / img.width, size / img.height);
  const drawWidth = img.width * scale;
  const drawHeight = img.height * scale;
  const offsetX = (size - drawWidth) / 2;
  const offsetY = (size - drawHeight) / 2;

  ctx.drawImage(img, offsetX, offsetY, drawWidth, drawHeight);
  return canvas.toDataURL("image/png");
}

async function resizeImageToSquare(file, size = TARGET_SIZE) {
  const img = await loadImageFromFile(file);
  const dataUrl = await renderSquarePreview(img, size);
  const blob = await (await fetch(dataUrl)).blob();

  const baseName = file.name.replace(/\.[^.]+$/, "") || "upload";
  return new File([blob], `${baseName}-256.png`, { type: "image/png" });
}

function App() {
  const [file, setFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState("");
  const [processedInputUrl, setProcessedInputUrl] = useState("");
  const [resultUrl, setResultUrl] = useState("");
  const [processedOutputUrl, setProcessedOutputUrl] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    if (!file) {
      setPreviewUrl("");
      setProcessedInputUrl("");
      return;
    }

    const url = URL.createObjectURL(file);
    setPreviewUrl(url);
    let cancelled = false;

    loadImageFromFile(file)
      .then((img) => renderSquarePreview(img, TARGET_SIZE))
      .then((dataUrl) => {
        if (!cancelled) setProcessedInputUrl(dataUrl);
      })
      .catch(() => {
        if (!cancelled) setProcessedInputUrl("");
      });

    return () => {
      cancelled = true;
      URL.revokeObjectURL(url);
    };
  }, [file]);

  useEffect(() => {
    if (!resultUrl) {
      setProcessedOutputUrl("");
      return;
    }

    let cancelled = false;
    loadImageFromUrl(resultUrl)
      .then((img) => renderSquarePreview(img, TARGET_SIZE))
      .then((dataUrl) => {
        if (!cancelled) setProcessedOutputUrl(dataUrl);
      })
      .catch(() => {
        if (!cancelled) setProcessedOutputUrl("");
      });

    return () => {
      cancelled = true;
    };
  }, [resultUrl]);

  const statusText = useMemo(() => {
    if (loading) return "Generating comic...";
    if (resultUrl) return "Comic ready";
    return "Upload a face to begin";
  }, [loading, resultUrl]);

  async function handleSubmit(event) {
    event.preventDefault();
    setError("");
    setResultUrl("");
    setProcessedOutputUrl("");

    if (!file) {
      setError("Please choose a face image first.");
      return;
    }

    setLoading(true);
    try {
      const resizedFile = await resizeImageToSquare(file, TARGET_SIZE);
      const data = new FormData();
      data.append("image", resizedFile);

      const res = await fetch("/generate", {
        method: "POST",
        body: data,
      });

      if (!res.ok) {
        const payload = await res.json().catch(() => ({}));
        throw new Error(payload.error || "Generation failed.");
      }

      const payload = await res.json();
      if (!payload.outputUrl) {
        throw new Error("No output returned from server.");
      }

      setResultUrl(payload.outputUrl);
    } catch (err) {
      setError(err.message || "Something went wrong.");
    } finally {
      setLoading(false);
    }
  }

  return e(
    "div",
    { className: "page" },
    e(
      "header",
      { className: "hero" },
      e("div", { className: "hero__content" },
        e("p", { className: "hero__eyebrow" }, "Face2Comic"),
        e("h1", { className: "hero__title" }, "Turn real faces into comic art."),
        e(
          "p",
          { className: "hero__subtitle" },
          "Upload a face image and the pix2pix generator will sketch a comic-style version in seconds."
        )
      ),
      e("div", { className: "hero__status" },
        e("span", { className: "status-pill" }, statusText)
      )
    ),
    e(
      "main",
      { className: "main" },
      e(
        "section",
        { className: "panel" },
        e("h2", { className: "panel__title" }, "Preprocessing Checklist"),
        e(
          "ol",
          { className: "checklist" },
          e(
            "li",
            { className: "checklist__item" },
            e("span", { className: "checklist__title" }, "Resize to 256x256"),
            e(
              "p",
              { className: "checklist__desc" },
              "Center-crop with cover scaling so the face fills the square input."
            )
          ),
          e(
            "li",
            { className: "checklist__item" },
            e("span", { className: "checklist__title" }, "Normalize to [-1, 1]"),
            e(
              "p",
              { className: "checklist__desc" },
              "Convert to float, divide by 255, then scale to match tanh output."
            )
          ),
          e(
            "li",
            { className: "checklist__item" },
            e("span", { className: "checklist__title" }, "Channel order"),
            e(
              "p",
              { className: "checklist__desc" },
              "Reorder to channels-first (C, H, W) for the PyTorch model."
            )
          )
        )
      ),
      e(
        "section",
        { className: "panel" },
        e("h2", { className: "panel__title" }, "Upload & Generate"),
        e(
          "form",
          { className: "form", onSubmit: handleSubmit },
          e(
            "label",
            { className: "file" },
            e("input", {
              type: "file",
              accept: "image/*",
              onChange: (event) => setFile(event.target.files[0] || null),
              disabled: loading,
            }),
            e("span", { className: "file__label" }, file ? file.name : "Choose a face image")
          ),
          e(
            "button",
            { className: "button", type: "submit", disabled: loading },
            loading ? "Generating..." : "Generate Comic"
          )
        ),
        error && e("p", { className: "error" }, error)
      ),
      e(
        "section",
        { className: "panel" },
        e("h2", { className: "panel__title" }, "Preview"),
        e(
          "div",
          { className: "grid" },
          e(
            "div",
            { className: "card" },
            e("h3", null, "Original"),
            e(
              "div",
              { className: "compare" },
              e(
                "div",
                { className: "compare__item" },
                e("span", { className: "compare__label" }, "Uploaded"),
                previewUrl
                  ? e("img", { src: previewUrl, alt: "Original preview" })
                  : e("div", { className: "placeholder" }, "No image yet")
              ),
              e(
                "div",
                { className: "compare__item" },
                e("span", { className: "compare__label" }, "Preprocessed"),
                processedInputUrl
                  ? e("img", { src: processedInputUrl, alt: "Preprocessed input" })
                  : e("div", { className: "placeholder" }, "No preprocessing yet")
              )
            )
          ),
          e(
            "div",
            { className: "card" },
            e("h3", null, "Comic"),
            e(
              "div",
              { className: "compare" },
              e(
                "div",
                { className: "compare__item" },
                e("span", { className: "compare__label" }, "Generated"),
                resultUrl
                  ? e("img", { src: resultUrl, alt: "Comic result" })
                  : loading
                    ? e(
                        "div",
                        { className: "inline-loader" },
                        e("div", { className: "inline-loader__spinner" }),
                        e("p", null, "Generating comic...")
                      )
                    : e("div", { className: "placeholder" }, "Waiting for output")
              ),
              e(
                "div",
                { className: "compare__item" },
                e("span", { className: "compare__label" }, "Preprocessed"),
                processedOutputUrl
                  ? e("img", { src: processedOutputUrl, alt: "Preprocessed output" })
                  : loading
                    ? e("div", { className: "placeholder" }, "Preparing preview...")
                    : e("div", { className: "placeholder" }, "No preprocessing yet")
              )
            )
          )
        )
      )
    ),
    loading && e("div", { className: "overlay" }, e("div", { className: "spinner" }))
  );
}

const root = ReactDOM.createRoot(document.getElementById("app"));
root.render(e(App));
