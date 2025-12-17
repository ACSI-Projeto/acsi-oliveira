import io
import json
import re
import uuid
from datetime import datetime, timezone

import easyocr
import numpy as np
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from kafka import KafkaProducer
from PIL import Image
from starlette.staticfiles import StaticFiles
from ultralytics.models import YOLO


class ANPR_V8:
    def __init__(self, model_path):
        self.model = YOLO(model_path)  # type: ignore

    def detect(self, img, threshold=0.3):
        results = self.model(img)
        plates = []
        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                conf = float(box.conf[0])
                if conf > threshold:
                    plates.append((x1, y1, x2, y2, conf))
        return plates


app = FastAPI()

# servir ficheiros estáticos em /static
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
async def read_index():
    return FileResponse("static/index.html")


# modelo YOLO de matrículas
detector = ANPR_V8("anpr_v8.pt")
reader = easyocr.Reader(["pt"])


def extrair_matricula_ocr(img: Image.Image):
    np_img = np.array(img)
    results = reader.readtext(np_img, allowlist=".-0123456789ABCDEFGHJKLMNPQRSTUVWXYZ")
    text = "".join([r[1] for r in results])

    # 1) limpar lixo
    text = text.strip().replace(" ", "").replace("]", "").replace("}", "")

    # 2) formatos PT típicos: AA-00-AA, 00-AA-00, 00-00-AA, AA-00-00
    padrao = r"([A-Z]{2}\d{2}[A-Z]{2}|\d{2}[A-Z]{2}\d{2}|\d{2}\d{2}[A-Z]{2}|[A-Z]{2}\d{2}\d{2})"
    match = re.search(padrao, text)

    matricula = match.group(1) if match else None
    return text, matricula


@app.post("/ocr")
async def ocr_endpoint(
    file: UploadFile = File(...),
    razaoPrimariaMulta: str = Form("ESTACIONAMENTO_PROIBIDO"),
    lugarId: str = Form("PRAÇA-001"),
    zonaTarifaria: str = Form("ZONA_A"),
    idFiscal: str = Form("FISCAL_1"),
):
    conteudo = await file.read()
    imagem = Image.open(io.BytesIO(conteudo)).convert("RGB")

    plates = detector.detect(imagem)
    if not plates:
        return JSONResponse({"error": "Nenhuma matrícula encontrada."})

    x1, y1, x2, y2, conf = plates[0]
    crop = imagem.crop((x1, y1, x2, y2))
    raw_text, matricula = extrair_matricula_ocr(crop)

    timestamp = datetime.now(timezone.utc).isoformat()

    multa = {
        "tipo": "multa",
        "idMulta": str(uuid.uuid4()),
        "matricula": matricula,
        "razaoPrimariaMulta": razaoPrimariaMulta,
        "lugarId": lugarId,
        "zonaTarifaria": zonaTarifaria,
        "dataHoraDeteccao": timestamp,
        "idFiscal": idFiscal,
        "origem": "APP_FISCAL",
        "tipoDeteccao": "OCR",
        "idEvidenciaPrimaria": None,
        "confianca": round(conf, 2),
    }

    producer = KafkaProducer(
        bootstrap_servers="localhost:29092",
        value_serializer=lambda v: json.dumps(v).encode(
            "utf-8"
        ),  # serializa o dict/objeto em JSON
    )
    producer.send("detalhes-multa", value=multa)
    producer.flush()

    return multa
