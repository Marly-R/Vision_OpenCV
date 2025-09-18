
// Función para cargar clasificadores en la memoria virtual de OpenCV.js
async function loadCascade(url, filename) {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`No se pudo cargar ${url}`);
  }
  const data = await response.arrayBuffer();
  const bytes = new Uint8Array(data);
  cv.FS_createDataFile('/', filename, bytes, true, false, false);
  const classifier = new cv.CascadeClassifier();
  classifier.load(filename);
  console.log(`✅ Clasificador ${filename} cargado`);
  return classifier;
}

// Inicialización de OpenCV
cv['onRuntimeInitialized'] = async () => {
  console.log("✅ OpenCV cargado y listo 🚀");

  // Cargar clasificadores Haar desde carpeta local /data
  const faceClassifier = await loadCascade('data/haarcascade_frontalface_default.xml', 'frontalface.xml');
  const eyeClassifier = await loadCascade('data/haarcascade_eye.xml', 'eye.xml');
  const mouthClassifier = await loadCascade('data/haarcascade_mcs_mouth.xml', 'mouth.xml');

  // Referencias al video y canvas
  const video = document.getElementById("camara");
  const canvas = document.getElementById("canvas");
  const ctx = canvas.getContext("2d");

  // Referencias a los spans del HTML
  const blinkSpan = document.getElementById("blinkCount");
  const mouthSpan = document.getElementById("mouthCount");
  const eyebrowSpan = document.getElementById("eyebrowCount");

  // Contadores
  let blinkCount = 0;
  let mouthCount = 0;
  let eyebrowCount = 0;

  // Estados previos
  let prevEyes = 0;
  let prevMouthOpen = false;
  let lastEyebrowTime = 0;

  // Configurar cámara
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ video: true });
    video.srcObject = stream;

    video.onloadedmetadata = () => {
      video.play();
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      requestAnimationFrame(detectar);
    };
  } catch (err) {
    console.error("❌ Error al acceder a la cámara:", err);
    return;
  }

  function detectar() {
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    let src = cv.imread(canvas);
    let gray = new cv.Mat();
    cv.cvtColor(src, gray, cv.COLOR_RGBA2GRAY, 0);

    // Detección de rostros
    let faces = new cv.RectVector();
    faceClassifier.detectMultiScale(gray, faces, 1.3, 5);

    for (let i = 0; i < faces.size(); i++) {
      let face = faces.get(i);

      // Dibujar rectángulo alrededor del rostro completo
      let p1 = new cv.Point(face.x, face.y);
      let p2 = new cv.Point(face.x + face.width, face.y + face.height);
      cv.rectangle(src, p1, p2, [255, 0, 0, 255]);

      // ROI general del rostro
      let roiGray = gray.roi(face);

      // --- Ojos y cejas (parte superior del rostro) ---
      let eyesROI = roiGray.roi(new cv.Rect(0, 0, face.width, face.height / 2));
      let eyes = new cv.RectVector();
      eyeClassifier.detectMultiScale(eyesROI, eyes);

      // Contador de parpadeo
      if (eyes.size() === 0 && prevEyes > 0) {
        blinkCount++;
        blinkSpan.textContent = blinkCount;
        console.log("👁️ Parpadeo detectado. Total:", blinkCount);
      }
      prevEyes = eyes.size();

      for (let j = 0; j < eyes.size(); j++) {
        let eye = eyes.get(j);
        let p1Eye = new cv.Point(face.x + eye.x, face.y + eye.y);
        let p2Eye = new cv.Point(face.x + eye.x + eye.width, face.y + eye.y + eye.height);
        cv.rectangle(src, p1Eye, p2Eye, [0, 255, 0, 255]);

        // Heurística cejas levantadas (arriba del ojo)
        if (eye.y < face.height * 0.2) { 
          let now = Date.now();
          if (now - lastEyebrowTime > 1000) { // 1 segundo entre detecciones
            eyebrowCount++;
            eyebrowSpan.textContent = eyebrowCount;
            console.log("⬆️ Cejas levantadas. Total:", eyebrowCount);
            lastEyebrowTime = now;
          }
        }
      }

      eyesROI.delete(); eyes.delete();

      // --- Boca (parte inferior del rostro) ---
      let mouthROI = roiGray.roi(new cv.Rect(0, face.height / 2, face.width, face.height / 2));
      let mouths = new cv.RectVector();
      mouthClassifier.detectMultiScale(mouthROI, mouths, 1.7, 11);

      let mouthOpen = mouths.size() > 0;
      if (mouthOpen && !prevMouthOpen) {
        mouthCount++;
        mouthSpan.textContent = mouthCount;
        console.log("👄 Boca abierta detectada. Total:", mouthCount);
      }
      prevMouthOpen = mouthOpen;

      for (let k = 0; k < mouths.size(); k++) {
        let mouth = mouths.get(k);
        let p1Mouth = new cv.Point(face.x + mouth.x, face.y + face.height / 2 + mouth.y);
        let p2Mouth = new cv.Point(face.x + mouth.x + mouth.width, face.y + face.height / 2 + mouth.y + mouth.height);
        cv.rectangle(src, p1Mouth, p2Mouth, [0, 0, 255, 255]);
      }

      roiGray.delete(); mouthROI.delete();
    }

    cv.imshow("canvas", src);
    src.delete(); gray.delete(); faces.delete();

    requestAnimationFrame(detectar);
  }
};
