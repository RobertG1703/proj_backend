"""
CameraFaceLandmarksGenerator - Gerador de dados de face landmarks com imagem mock

Resumo:
Gera dados realistas de face landmarks (478 pontos), gaze tracking, Eye Aspect Ratio
e blink detection com imagem visual coordenada. Simula padrões naturais de atenção,
sonolência e movimento facial. A imagem mock é desenhada baseada nos landmarks gerados
e coordena blinks visuais com valores EAR baixos.
"""

import logging
import numpy as np
import base64
import io
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from enum import Enum
from PIL import Image, ImageDraw

from app.core import settings

class CameraAnomalyType(Enum):
    """Tipos de anomalias específicas de dados de câmera"""
    NORMAL = "normal"
    LOW_BLINK_RATE = "low_blink_rate"                    # Poucas piscadelas (sonolência)
    HIGH_BLINK_RATE = "high_blink_rate"                  # Muitas piscadelas (stress)
    POOR_DETECTION = "poor_detection"                    # Qualidade de deteção baixa
    GAZE_DRIFT = "gaze_drift"                           # Olhar muito desviado
    EXCESSIVE_MOVEMENT = "excessive_movement"            # Movimento excessivo da cabeça
    DISTRACTED_GAZE = "distracted_gaze"                 # Padrão de olhar errático

class AttentionPattern(Enum):
    """Padrões de atenção simulados"""
    FOCUSED = "focused"                                  # Focado na estrada
    DISTRACTED = "distracted"                           # Distraído
    DROWSY = "drowsy"                                    # Sonolento
    ALERT = "alert"                                      # Muito alerta
    CHECKING_MIRRORS = "checking_mirrors"               # A verificar espelhos
    LOOKING_ASIDE = "looking_aside"                     # A olhar para o lado
    READING_DASHBOARD = "reading_dashboard"             # A ler dashboard

class CameraFaceLandmarksGenerator:
    """Gerador de dados de face landmarks para tópico Camera_FaceLandmarks"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Configurações Camera do settings
        self.cameraConfig = settings.signals.cameraConfig
        self.mockCameraConfig = settings.mockZeromq.generatorBaseConfig["camera"]

        self.landmarksConfig = self.cameraConfig["faceLandmarks"]
        self.landmarkRanges = self.mockCameraConfig["landmarkRanges"]
        self.anatomyPositions = self.mockCameraConfig["anatomyPositions"]

        # Padrões e anomalias
        self.attentionPatterns = self.mockCameraConfig["attentionPatterns"]
        self.anomalyTypes = self.mockCameraConfig["anomalyTypes"]

        self.gazeConfig = self.cameraConfig["gaze"]
        self.blinkConfig = self.cameraConfig["blink_rate"]
        self.earConfig = self.cameraConfig["ear"]
        self.mockConfig = settings.mockZeromq
        
        # Configurações de geração
        self.fps = self.landmarksConfig["fps"]                                    # 0.5Hz
        self.frameDuration = 1.0 / self.fps                                      # 2s por frame
        self.landmarksCount = self.landmarksConfig["landmarksCount"]             # 478
        
        # Configurações de anomalias
        self.anomalyConfig = self.mockConfig.anomalyInjection
        self.anomalyChance = self.anomalyConfig["topicChances"]["Camera_FaceLandmarks"]  # 1%
        
        # Estado interno do gerador
        self.currentTimestamp = 0.0
        self.frameCounter = 0
        self.lastAnomalyTime = 0.0
        self.currentAnomalyType = CameraAnomalyType.NORMAL
        self.anomalyDuration = 0.0
        self.anomalyStartTime = 0.0
        
        # Estado de atenção simulada
        self.currentAttentionPattern = AttentionPattern.FOCUSED
        self.patternStartTime = 0.0
        self.patternDuration = 0.0
        
        # Estado facial
        self.currentEar = 0.3                           # Eye Aspect Ratio atual
        self.isBlinking = False                         # Estado atual de blink
        self.blinkStartTime = 0.0                       # Timestamp do início do blink
        self.blinkDuration = 0.15                       # Duração típica do blink (150ms)
        self.lastBlinkTime = 0.0                        # Último blink registado
        self.blinkCounter = 0                           # Contador total de blinks
        self.recentBlinkTimes: List[float] = []         # Timestamps dos últimos blinks
        
        # Gaze tracking
        self.currentGazeVector = {"dx": 0.0, "dy": 0.0}  # Direção atual do olhar
        self.gazeTarget = {"dx": 0.0, "dy": 0.0}         # Target para suavização
        self.gazeSmoothingFactor = self.mockCameraConfig["naturalMovement"]["gazeSmoothingFactor"]
        
        # Face landmarks base (template facial normalizado)
        self.baseLandmarks = self._generateBaseFaceLandmarks()
        self.currentLandmarks = self.baseLandmarks.copy()
        
        # Parâmetros de movimento facial
        self.headPosition = np.array(self.anatomyPositions["faceCenter"])   # Centro da face normalizado
        self.headRotation = np.array([0.0, 0.0, 0.0])   # Rotação da cabeça (pitch, yaw, roll)
        self.microMovementPhase = 0.0                    # Fase para micro-movimentos
        
        # Configurações de imagem
        self.imageSize = tuple(self.mockCameraConfig["mockImage"]["size"])                      # Tamanho da imagem mock
        self.imageQuality = self.mockCameraConfig["mockImage"]["quality"]                       # Qualidade JPEG
        
        self.logger.info(f"CameraFaceLandmarksGenerator initialized - {self.fps}Hz, {self.landmarksCount} landmarks")
    
    def generateFrame(self, baseTimestamp: Optional[float] = None) -> Dict[str, Any]:
        """
        Gera um frame de dados de câmera (landmarks + imagem).
        
        Args:
            baseTimestamp: Timestamp base para o frame (usa interno se None)
            
        Returns:
            Dict com dados de câmera para formatação
        """
        
        if baseTimestamp is not None:
            self.currentTimestamp = baseTimestamp
        
        try:
            # Atualizar padrão de atenção
            self._updateAttentionPattern()
            
            # Verificar se deve injetar anomalia
            self._updateAnomalyState()
            
            # Atualizar estado de blink
            self._updateBlinkState()
            
            # Atualizar gaze tracking
            self._updateGazeTracking()
            
            # Atualizar movimento da cabeça
            self._updateHeadMovement()
            
            # Gerar landmarks baseados no estado atual
            landmarks = self._generateCurrentLandmarks()
            
            # Calcular valores derivados
            ear = self._calculateEAR(landmarks)
            blinkRate = self._calculateBlinkRate()
            
            # Gerar imagem mock coordenada
            frameImage = self._generateMockImage(landmarks, ear)
            
            # Avançar timestamp para próximo frame
            self.currentTimestamp += self.frameDuration
            self.frameCounter += 1
            
            result = {
                "landmarks": landmarks.tolist(),
                "gaze_vector": self.currentGazeVector.copy(),
                "ear": ear,
                "blink_rate": blinkRate,
                "blink_counter": self.blinkCounter,
                "frame_b64": frameImage,
                "frameTimestamp": self.currentTimestamp - self.frameDuration,
                "anomalyType": self.currentAnomalyType.value,
                "attentionPattern": self.currentAttentionPattern.value,
                "isBlinking": self.isBlinking,
                "frameNumber": self.frameCounter
            }
            
            self.logger.debug(f"Generated camera frame {self.frameCounter}: EAR={ear:.3f}, BlinkRate={blinkRate:.1f}, BlinkCounter={self.blinkCounter}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error generating camera frame: {e}")
            raise
    
    def _generateBaseFaceLandmarks(self) -> np.ndarray:
        """
        Gera template base de 478 landmarks faciais normalizados.
        Simplificado mas anatomicamente plausível.
        
        Returns:
            Array (478, 3) com landmarks base
        """
        
        landmarks = np.zeros((478, 3))

        # Usar ranges da config
        ranges = self.landmarkRanges
        
        # Face outline
        start, end = ranges["faceOutline"]
        landmarks[start:end+1] = self._generateFaceOutline()
        
        # Eyebrows  
        start, end = ranges["eyebrows"]
        landmarks[start:end+1] = self._generateEyebrows()
        
        # Eyes
        start, end = ranges["leftEye"]
        landmarks[start:end+1] = self._generateLeftEye()
        
        start, end = ranges["rightEye"] 
        landmarks[start:end+1] = self._generateRightEye()
    
        # Nariz
        start, end = ranges["nose"]
        landmarks[start:end+1] = self._generateNose()
        
        # Mouth
        start, end = ranges["mouth"]
        landmarks[start:end+1] = self._generateMouth()
        
        # Preencher landmarks restantes com distribuição facial plausível
        self._fillRemainingLandmarks(landmarks)
        
        return landmarks
    
    def _generateFaceOutline(self) -> np.ndarray:
        """Gera contorno facial oval"""
        ranges = self.landmarkRanges["faceOutline"]
        numPoints = ranges[1] - ranges[0] + 1
        outline = np.zeros((numPoints, 3))
        
        faceWidth = self.anatomyPositions["faceWidth"]
        faceHeight = self.anatomyPositions["faceHeight"]
        faceCenter = self.anatomyPositions["faceCenter"]
        
        # Contorno oval baseado nas configurações
        for i in range(numPoints):
            angle = i * 2 * np.pi / (numPoints - 1)  # 0 a 2π
            x = faceCenter[0] + (faceWidth / 2) * np.cos(angle + np.pi/2)
            y = faceCenter[1] - faceHeight / 2 + (faceHeight / 2) * (1 - np.cos(angle))
            z = faceCenter[2]  # Plano frontal
            outline[i] = [x, y, z]
        
        return outline
    
    def _generateEyebrows(self) -> np.ndarray:
        """Gera sobrancelhas"""
        ranges = self.landmarkRanges["eyebrows"]
        numPoints = ranges[1] - ranges[0] + 1
        eyebrows = np.zeros((numPoints, 3))
        
        leftEyeCenter = self.anatomyPositions["leftEyeCenter"]
        rightEyeCenter = self.anatomyPositions["rightEyeCenter"]
        eyeWidth = self.anatomyPositions["eyeWidth"]
        
        pointsPerEyebrow = numPoints // 2
        
        # Sobrancelha esquerda
        for i in range(pointsPerEyebrow):
            x = leftEyeCenter[0] - eyeWidth/2 + i * (eyeWidth / (pointsPerEyebrow - 1))
            y = leftEyeCenter[1] - 0.05 - 0.02 * np.sin(i * np.pi / (pointsPerEyebrow - 1))  # Curva ligeira
            eyebrows[i] = [x, y, 0.0]
        
        # Sobrancelha direita
        for i in range(pointsPerEyebrow):
            x = rightEyeCenter[0] + eyeWidth/2 - i * (eyeWidth / (pointsPerEyebrow - 1))
            y = rightEyeCenter[1] - 0.05 - 0.02 * np.sin(i * np.pi / (pointsPerEyebrow - 1))
            eyebrows[pointsPerEyebrow + i] = [x, y, 0.0]
        
        return eyebrows
    
    def _generateLeftEye(self) -> np.ndarray:
        """Gera landmarks do olho esquerdo"""
        ranges = self.landmarkRanges["leftEye"]
        numPoints = ranges[1] - ranges[0] + 1
        leftEye = np.zeros((numPoints, 3))
        
        centerX, centerY, centerZ = self.anatomyPositions["leftEyeCenter"]
        width = self.anatomyPositions["eyeWidth"]
        height = self.anatomyPositions["eyeHeight"]
        
        # Pontos do olho distribuídos em forma amendoada
        for i in range(numPoints):
            angle = i * 2 * np.pi / numPoints
            x = centerX + (width/2) * np.cos(angle)
            y = centerY + (height/2) * np.sin(angle)
            z = centerZ
            leftEye[i] = [x, y, z]
        
        return leftEye
    
    def _generateRightEye(self) -> np.ndarray:
        """Gera landmarks do olho direito (simétrico ao esquerdo)"""
        ranges = self.landmarkRanges["rightEye"]
        numPoints = ranges[1] - ranges[0] + 1
        rightEye = np.zeros((numPoints, 3))
        
        centerX, centerY, centerZ = self.anatomyPositions["rightEyeCenter"]
        width = self.anatomyPositions["eyeWidth"]
        height = self.anatomyPositions["eyeHeight"]
        
        # Pontos do olho distribuídos em forma amendoada
        for i in range(numPoints):
            angle = i * 2 * np.pi / numPoints
            x = centerX + (width/2) * np.cos(angle)
            y = centerY + (height/2) * np.sin(angle)
            z = centerZ
            rightEye[i] = [x, y, z]
        
        return rightEye
    
    def _generateNose(self) -> np.ndarray:
        """Gera landmarks do nariz"""
        ranges = self.landmarkRanges["nose"]
        numPoints = ranges[1] - ranges[0] + 1
        nose = np.zeros((numPoints, 3))
        
        noseCenter = self.anatomyPositions["noseCenter"]
        noseTip = self.anatomyPositions["noseTip"]
        
        bridgePoints = numPoints // 2
        nostrilPoints = numPoints - bridgePoints
        
        # Ponte do nariz
        for i in range(bridgePoints):
            t = i / (bridgePoints - 1) if bridgePoints > 1 else 0
            x = noseCenter[0]
            y = noseCenter[1] + t * (noseTip[1] - noseCenter[1])
            z = noseCenter[2] + t * (noseTip[2] - noseCenter[2])
            nose[i] = [x, y, z]
        
        # Narinas
        for i in range(nostrilPoints):
            side = -1 if i < nostrilPoints // 2 else 1
            x = noseTip[0] + side * 0.02
            y = noseTip[1] + 0.01 * (i % (nostrilPoints // 2))
            z = noseTip[2]
            nose[bridgePoints + i] = [x, y, z]
        
        return nose
    
    def _generateMouth(self) -> np.ndarray:
        """Gera landmarks da boca"""
        ranges = self.landmarkRanges["mouth"]
        numPoints = ranges[1] - ranges[0] + 1
        mouth = np.zeros((numPoints, 3))
        
        centerX, centerY, centerZ = self.anatomyPositions["mouthCenter"]
        width = self.anatomyPositions["mouthWidth"]
        height = self.anatomyPositions["mouthHeight"]
        
        outerPoints = numPoints // 2
        innerPoints = numPoints - outerPoints
        
        # Contorno exterior da boca
        for i in range(outerPoints):
            angle = i * 2 * np.pi / outerPoints
            x = centerX + (width/2) * np.cos(angle)
            y = centerY + (height/2) * np.sin(angle) * 0.6  # Forma oval
            mouth[i] = [x, y, centerZ]
        
        # Contorno interior
        for i in range(innerPoints):
            angle = i * 2 * np.pi / innerPoints
            x = centerX + (width/3) * np.cos(angle)
            y = centerY + (height/3) * np.sin(angle) * 0.4
            mouth[outerPoints + i] = [x, y, centerZ]
        
        return mouth
    
    def _fillRemainingLandmarks(self, landmarks: np.ndarray):
        """Preenche landmarks restantes com distribuição facial plausível"""
        
        ranges = self.landmarkRanges["remaining"]
        startIdx = ranges[0]
        endIdx = ranges[1]
        
        faceWidth = self.anatomyPositions["faceWidth"]
        faceHeight = self.anatomyPositions["faceHeight"]
        faceCenter = self.anatomyPositions["faceCenter"]
        
        # Para landmarks não definidos explicitamente, distribuir pela face
        for i in range(startIdx, endIdx + 1):
            # Distribuir aleatoriamente mas dentro da região facial
            x = faceCenter[0] + np.random.uniform(-faceWidth/2, faceWidth/2)
            y = faceCenter[1] + np.random.uniform(-faceHeight/2, faceHeight/2)
            z = np.random.uniform(-0.02, 0.02)
            landmarks[i] = [x, y, z]
    
    def _updateAttentionPattern(self):
        """Atualiza padrão de atenção baseado em probabilidades e timing"""
        
        currentTime = self.currentTimestamp
        
        # Verificar se deve mudar padrão
        if currentTime - self.patternStartTime >= self.patternDuration:
            # Escolher novo padrão baseado em probabilidades
            patterns = []
            weights = []
            
            for patternName, config in self.attentionPatterns.items():
                patterns.append(AttentionPattern(patternName))
                weights.append(config["probability"])
            
            self.currentAttentionPattern = np.random.choice(patterns, p=weights)
            
            self.patternStartTime = currentTime
            
            # Duração do padrão baseada na configuração
            patternConfig = self.attentionPatterns[self.currentAttentionPattern.value]
            durationRange = patternConfig["durationRange"]
            self.patternDuration = np.random.uniform(durationRange[0], durationRange[1])
            
            self.logger.debug(f"Attention pattern changed to: {self.currentAttentionPattern.value} for {self.patternDuration:.1f}s")
    
    def _updateBlinkState(self):
        """Atualiza estado de piscadelas baseado em padrões naturais"""
        
        currentTime = self.currentTimestamp
        
        # Se estiver a piscar, verificar se deve terminar
        if self.isBlinking:
            if currentTime - self.blinkStartTime >= self.blinkDuration:
                self.isBlinking = False
                self.logger.debug(f"Blink ended at {currentTime:.3f}s")
            return
        
        # Calcular probabilidade de blink baseada no padrão de atenção
        baseProbability = 0.02  # 2% chance por frame (0.5Hz)
        
        patternConfig = self.attentionPatterns[self.currentAttentionPattern.value]
        blinkMultiplier = patternConfig["blinkMultiplier"]
        blinkProbability = baseProbability * blinkMultiplier
        
        # Evitar blinks muito próximos
        minBlinkInterval = self.mockCameraConfig["naturalMovement"]["minBlinkInterval"]
        timeSinceLastBlink = currentTime - self.lastBlinkTime
        if timeSinceLastBlink < minBlinkInterval:
            blinkProbability *= 0.1
        
        # Decidir se deve piscar
        if np.random.random() < blinkProbability:
            self.isBlinking = True
            self.blinkStartTime = currentTime
            self.lastBlinkTime = currentTime
            self.blinkCounter += 1
            
            # Duração do blink baseada na configuração
            blinkDurationRange = self.mockCameraConfig["naturalMovement"]["blinkDurationRange"]
            self.blinkDuration = np.random.uniform(blinkDurationRange[0], blinkDurationRange[1])
            
            # Adicionar ao histórico de blinks
            self.recentBlinkTimes.append(currentTime)
            
            # Manter apenas blinks dos últimos 60 segundos
            cutoffTime = currentTime - 60.0
            self.recentBlinkTimes = [t for t in self.recentBlinkTimes if t > cutoffTime]
            
            self.logger.debug(f"Blink started at {currentTime:.3f}s (duration: {self.blinkDuration:.3f}s)")
    
    def _updateGazeTracking(self):
        """Atualiza direção do olhar baseada no padrão de atenção"""
        
        # Obter configuração do padrão atual
        patternConfig = self.attentionPatterns[self.currentAttentionPattern.value]
        gazeCenter = patternConfig["gazeCenter"]
        gazeVariation = patternConfig["gazeVariation"]
        
        # Definir target do gaze baseado no padrão atual
        self.gazeTarget["dx"] = gazeCenter[0] + np.random.normal(0, gazeVariation)
        self.gazeTarget["dy"] = gazeCenter[1] + np.random.normal(0, gazeVariation)
        
        # Suavizar movimento do gaze
        dxDiff = self.gazeTarget["dx"] - self.currentGazeVector["dx"]
        dyDiff = self.gazeTarget["dy"] - self.currentGazeVector["dy"]
        
        self.currentGazeVector["dx"] += dxDiff * self.gazeSmoothingFactor
        self.currentGazeVector["dy"] += dyDiff * self.gazeSmoothingFactor
        
        # Clipar para range válido
        self.currentGazeVector["dx"] = np.clip(self.currentGazeVector["dx"], -1.0, 1.0)
        self.currentGazeVector["dy"] = np.clip(self.currentGazeVector["dy"], -1.0, 1.0)
    
    def _updateHeadMovement(self):
        """Atualiza movimento subtil da cabeça"""
        
        # Micro-movimentos naturais
        self.microMovementPhase += 0.1
        
        # Configurações de movimento natural
        naturalMovement = self.mockCameraConfig["naturalMovement"]
        headMovementStd = naturalMovement["headMovementStd"]
        microMovementAmplitude = naturalMovement["microMovementAmplitude"]
        headPositionLimits = naturalMovement["headPositionLimits"]
        
        # Variação ligeira na posição da cabeça
        headNoise = np.random.normal(0, headMovementStd, 3)
        microMovement = np.array([
            microMovementAmplitude * np.sin(self.microMovementPhase),
            microMovementAmplitude * np.cos(self.microMovementPhase * 1.3),
            microMovementAmplitude * np.sin(self.microMovementPhase * 0.7)
        ])
        
        self.headPosition += headNoise + microMovement
        
        # Manter dentro de limites razoáveis
        self.headPosition = np.clip(self.headPosition, headPositionLimits["min"], headPositionLimits["max"])
    
    def _generateCurrentLandmarks(self) -> np.ndarray:
        """Gera landmarks atuais baseados no estado facial"""
        
        # Começar com landmarks base
        landmarks = self.baseLandmarks.copy()
        
        # Aplicar movimento da cabeça (translação ligeira)
        faceCenter = np.array(self.anatomyPositions["faceCenter"])
        headMovement = self.headPosition - faceCenter
        landmarks += headMovement
        
        # Aplicar efeito do gaze nos olhos
        self._applyGazeToEyes(landmarks)
        
        # Aplicar efeito do blink nos olhos
        if self.isBlinking:
            self._applyBlinkToEyes(landmarks)
        
        # Adicionar variação natural muito ligeira
        naturalVariationStd = self.mockCameraConfig["naturalMovement"]["naturalVariationStd"]
        naturalVariation = np.random.normal(0, naturalVariationStd, landmarks.shape)
        landmarks += naturalVariation
        
        # Aplicar anomalias se ativas
        if self.currentAnomalyType != CameraAnomalyType.NORMAL:
            landmarks = self._applyAnomalies(landmarks)
        
        # Garantir que landmarks ficam normalizados
        landmarks = np.clip(landmarks, 0.0, 1.0)
        
        return landmarks
    
    def _applyGazeToEyes(self, landmarks: np.ndarray):
        """Aplica efeito do gaze aos landmarks dos olhos"""
        
        gazeShift = np.array([self.currentGazeVector["dx"] * 0.01, self.currentGazeVector["dy"] * 0.01, 0])
        
        # Aplicar aos olhos usando os ranges configurados
        leftEyeRange = self.landmarkRanges["leftEye"]
        rightEyeRange = self.landmarkRanges["rightEye"]
        
        landmarks[leftEyeRange[0]:leftEyeRange[1]+1] += gazeShift
        landmarks[rightEyeRange[0]:rightEyeRange[1]+1] += gazeShift
    
    def _applyBlinkToEyes(self, landmarks: np.ndarray):
        """Aplica efeito do blink aos landmarks dos olhos"""
        
        # Durante blink, reduzir height dos olhos
        blinkIntensity = 0.02
        
        leftEyeRange = self.landmarkRanges["leftEye"]
        rightEyeRange = self.landmarkRanges["rightEye"]
        
        # Aplicar efeito aos pontos dos olhos
        for i in range(leftEyeRange[0], leftEyeRange[1] + 1):
            if i < len(landmarks):
                landmarks[i, 1] += blinkIntensity * np.sin((i - leftEyeRange[0]) * np.pi / (leftEyeRange[1] - leftEyeRange[0]))
        
        for i in range(rightEyeRange[0], rightEyeRange[1] + 1):
            if i < len(landmarks):
                landmarks[i, 1] += blinkIntensity * np.sin((i - rightEyeRange[0]) * np.pi / (rightEyeRange[1] - rightEyeRange[0]))
    
    def _calculateEAR(self, landmarks: np.ndarray) -> float:
        """Calcula Eye Aspect Ratio baseado nos landmarks dos olhos"""
        
        if self.isBlinking:
            # Durante blink, EAR muito baixo
            return np.random.uniform(0.05, 0.12)
        else:
            # EAR normal baseado no padrão de atenção
            patternConfig = self.attentionPatterns[self.currentAttentionPattern.value]
            baseEar = patternConfig["earBase"]
            
            # Adicionar variação natural
            ear = baseEar + np.random.normal(0, 0.03)
            return np.clip(ear, 0.1, 0.45)
    
    def _calculateBlinkRate(self) -> float:
        """Calcula taxa de piscadelas em blinks por minuto"""
        
        currentTime = self.currentTimestamp
        
        # Contar blinks nos últimos 60 segundos
        cutoffTime = currentTime - 60.0
        recentBlinks = [t for t in self.recentBlinkTimes if t > cutoffTime]
        
        # Calcular rate (blinks por minuto)
        blinkRate = len(recentBlinks)
        
        # Ajustar baseado no padrão de atenção
        patternConfig = self.attentionPatterns[self.currentAttentionPattern.value]
        blinkMultiplier = patternConfig["blinkMultiplier"]
        blinkRate *= blinkMultiplier
        
        return max(0.0, blinkRate)
    
    def _generateMockImage(self, landmarks: np.ndarray, ear: float) -> str:
        """
        Gera imagem mock da face baseada nos landmarks.
        
        Args:
            landmarks: Array de landmarks faciais
            ear: Eye Aspect Ratio atual
            
        Returns:
            Imagem encoded em base64
        """
        
        try:
            # Configurações da imagem
            mockImageConfig = self.mockCameraConfig["mockImage"]
            backgroundColor = mockImageConfig["backgroundColor"]
            colors = mockImageConfig["colors"]
            lineWidths = mockImageConfig["lineWidths"]
            # Criar imagem
            img = Image.new('RGB', self.imageSize, color=backgroundColor)
            draw = ImageDraw.Draw(img)

            
            # Converter landmarks normalizados para pixels
            imgWidth, imgHeight = self.imageSize
            pixelLandmarks = landmarks.copy()
            pixelLandmarks[:, 0] *= imgWidth   # X para pixels
            pixelLandmarks[:, 1] *= imgHeight  # Y para pixels
            
            # Desenhar contorno facial
            self._drawFaceOutline(draw, pixelLandmarks, colors, lineWidths)

            
            # Desenhar features faciais
            self._drawEyebrows(draw, pixelLandmarks, colors, lineWidths)
            self._drawNose(draw, pixelLandmarks, colors, lineWidths)
            self._drawMouth(draw, pixelLandmarks, colors, lineWidths)
            
            # Desenhar olhos (coordenados com EAR)
            self._drawEyes(draw, pixelLandmarks, ear, colors, lineWidths)

            # Adicionar pupils baseadas no gaze
            self._drawPupils(draw, pixelLandmarks, ear, colors)

            # Converter para base64
            buffer = io.BytesIO()
            img.save(buffer, format='JPEG', quality=self.imageQuality)
            imgBytes = buffer.getvalue()
            
            return base64.b64encode(imgBytes).decode('utf-8')
            
        except Exception as e:
            self.logger.error(f"Error generating mock image: {e}")
            # Retornar placeholder em caso de erro
            return "mock_camera_frame_error"
    
    def _drawFaceOutline(self, draw: ImageDraw.Draw, landmarks: np.ndarray, colors: Dict, lineWidths: Dict):
        """Desenha contorno facial baseado nos landmarks"""
        
        # Usar range configurado para contorno
        faceOutlineRange = self.landmarkRanges["faceOutline"]
        facePoints = [(landmarks[i, 0], landmarks[i, 1]) for i in range(faceOutlineRange[0], faceOutlineRange[1] + 1)]
        
        if len(facePoints) > 2:
            draw.polygon(facePoints, outline=colors["faceOutline"], width=lineWidths["faceOutline"])
    
    def _drawEyebrows(self, draw: ImageDraw.Draw, landmarks: np.ndarray, colors: Dict, lineWidths: Dict):
        """Desenha sobrancelhas"""
        
        # Usar ranges configurados para sobrancelhas
        leftBrowRange = self.landmarkRanges["leftEyebrow"]
        rightBrowRange = self.landmarkRanges["rightEyebrow"]

        # Sobrancelha esquerda
        leftBrow = [(landmarks[i, 0], landmarks[i, 1]) for i in range(leftBrowRange[0], leftBrowRange[1] + 1)]
        if len(leftBrow) > 1:
            for i in range(len(leftBrow) - 1):
                draw.line([leftBrow[i], leftBrow[i+1]], fill=colors["eyebrows"], width=lineWidths["eyebrows"])

        # Sobrancelha direita
        rightBrow = [(landmarks[i, 0], landmarks[i, 1]) for i in range(rightBrowRange[0], rightBrowRange[1] + 1)]
        if len(rightBrow) > 1:
            for i in range(len(rightBrow) - 1):
                draw.line([rightBrow[i], rightBrow[i+1]], fill=colors["eyebrows"], width=lineWidths["eyebrows"])

    def _drawNose(self, draw: ImageDraw.Draw, landmarks: np.ndarray, colors: Dict, lineWidths: Dict):
        """Desenha nariz"""
        
        # Usar ranges configurados para nariz
        noseBridgeRange = self.landmarkRanges["noseBridge"]
        noseNostrilsRange = self.landmarkRanges["noseNostrils"]
        
        # Ponte do nariz
        bridgePoints = [(landmarks[i, 0], landmarks[i, 1]) for i in range(noseBridgeRange[0], noseBridgeRange[1] + 1)]
        
        if len(bridgePoints) >= 2:
            # Desenhar ponte
            for i in range(len(bridgePoints) - 1):
                draw.line([bridgePoints[i], bridgePoints[i+1]], fill=colors["nose"], width=lineWidths["nose"])
        
        # Narinas
        nostrilPoints = [(landmarks[i, 0], landmarks[i, 1]) for i in range(noseNostrilsRange[0], noseNostrilsRange[1] + 1)]
        if len(nostrilPoints) >= 2:
            nostrilSize = 2
            for point in nostrilPoints[:2]:  # Desenhar apenas as primeiras duas narinas
                draw.ellipse([point[0]-nostrilSize, point[1]-nostrilSize, 
                             point[0]+nostrilSize, point[1]+nostrilSize], 
                             outline=colors["nose"])
    
    def _drawMouth(self, draw: ImageDraw.Draw, landmarks: np.ndarray, colors: Dict, lineWidths: Dict):
        """Desenha boca"""
        
        # Usar range configurado para boca
        outerMouthRange = self.landmarkRanges["outerMouth"]
        mouthPoints = [(landmarks[i, 0], landmarks[i, 1]) for i in range(outerMouthRange[0], outerMouthRange[1] + 1)]
        
        if len(mouthPoints) >= 3:
            # Desenhar contorno exterior
            mouthPoints.append(mouthPoints[0])  # Fechar polygon
            draw.polygon(mouthPoints, outline=colors["mouth"], width=lineWidths["mouth"])
    
    def _drawEyes(self, draw: ImageDraw.Draw, landmarks: np.ndarray, ear: float, colors: Dict, lineWidths: Dict):
        """Desenha olhos coordenados com EAR"""
        
        # Usar ranges configurados para olhos
        leftEyeRange = self.landmarkRanges["leftEye"]
        rightEyeRange = self.landmarkRanges["rightEye"]
        
        leftEyePoints = [(landmarks[i, 0], landmarks[i, 1]) for i in range(leftEyeRange[0], leftEyeRange[1] + 1)]
        rightEyePoints = [(landmarks[i, 0], landmarks[i, 1]) for i in range(rightEyeRange[0], rightEyeRange[1] + 1)]
        
        # Determinar se olhos estão fechados baseado em EAR
        eyesClosed = ear < 0.15 or self.isBlinking
        
        if eyesClosed:
            # Desenhar olhos fechados (linhas horizontais)
            if len(leftEyePoints) >= 2:
                draw.line([leftEyePoints[0], leftEyePoints[-1]], fill=colors["eyeClosed"], width=lineWidths["eyeClosed"])
            if len(rightEyePoints) >= 2:
                draw.line([rightEyePoints[0], rightEyePoints[-1]], fill=colors["eyeClosed"], width=lineWidths["eyeClosed"])
        else:
            # Desenhar olhos abertos (elipses)
            if len(leftEyePoints) >= 3:
                # Calcular bounding box do olho esquerdo
                leftX = [p[0] for p in leftEyePoints]
                leftY = [p[1] for p in leftEyePoints]
                leftBbox = [min(leftX), min(leftY), max(leftX), max(leftY)]
                draw.ellipse(leftBbox, outline=colors["eyeOpen"], fill=colors["eyeOpen"], width=lineWidths["eyeOpen"])
            
            if len(rightEyePoints) >= 3:
                # Calcular bounding box do olho direito
                rightX = [p[0] for p in rightEyePoints]
                rightY = [p[1] for p in rightEyePoints]
                rightBbox = [min(rightX), min(rightY), max(rightX), max(rightY)]
                draw.ellipse(rightBbox, outline=colors["eyeOpen"], fill=colors["eyeOpen"], width=lineWidths["eyeOpen"])
    
    def _drawPupils(self, draw: ImageDraw.Draw, landmarks: np.ndarray, ear: float, colors: Dict):
        """Desenha pupilas baseadas no gaze direction"""
        
        # Só desenhar pupilas se olhos estiverem abertos
        if ear < 0.15 or self.isBlinking:
            return
        
        # Calcular posição das pupilas baseada no gaze
        gazeOffsetX = self.currentGazeVector["dx"] * 3  # Pixels
        gazeOffsetY = self.currentGazeVector["dy"] * 2  # Pixels
        
        # Usar centros dos olhos configurados
        leftEyeCenter = self.anatomyPositions["leftEyeCenter"]
        rightEyeCenter = self.anatomyPositions["rightEyeCenter"]
        
        # Converter para pixels
        imgWidth, imgHeight = self.imageSize
        leftEyeCenterPx = [leftEyeCenter[0] * imgWidth, leftEyeCenter[1] * imgHeight]
        rightEyeCenterPx = [rightEyeCenter[0] * imgWidth, rightEyeCenter[1] * imgHeight]
        
        # Pupila esquerda
        pupilX = leftEyeCenterPx[0] + gazeOffsetX
        pupilY = leftEyeCenterPx[1] + gazeOffsetY
        draw.ellipse([pupilX-2, pupilY-2, pupilX+2, pupilY+2], fill=colors["pupil"])
        
        # Pupila direita
        pupilX = rightEyeCenterPx[0] + gazeOffsetX
        pupilY = rightEyeCenterPx[1] + gazeOffsetY
        draw.ellipse([pupilX-2, pupilY-2, pupilX+2, pupilY+2], fill=colors["pupil"])
    
    def _applyAnomalies(self, landmarks: np.ndarray) -> np.ndarray:
        """Aplica anomalias específicas aos landmarks"""
        
        anomalyConfig = self.anomalyTypes[self.currentAnomalyType.value.lower()]
        
        if self.currentAnomalyType == CameraAnomalyType.EXCESSIVE_MOVEMENT:
            # Movimento excessivo - usar configuração centralizada
            movementMultiplier = anomalyConfig.get("movementMultiplier", 10.0)
            naturalVariationStd = self.mockCameraConfig["naturalMovement"]["naturalVariationStd"]
            movement = np.random.normal(0, naturalVariationStd * movementMultiplier, landmarks.shape)
            landmarks += movement
            
        elif self.currentAnomalyType == CameraAnomalyType.POOR_DETECTION:
            # Deteção pobre - usar multiplicador de ruído configurado
            noiseMultiplier = anomalyConfig.get("noiseMultiplier", 5.0)
            naturalVariationStd = self.mockCameraConfig["naturalMovement"]["naturalVariationStd"]
            noise = np.random.normal(0, naturalVariationStd * noiseMultiplier, landmarks.shape)
            landmarks += noise
            
        elif self.currentAnomalyType == CameraAnomalyType.GAZE_DRIFT:
            # Gaze errático - usar força configurada
            gazeForce = anomalyConfig.get("gazeForce", 0.9)
            self.currentGazeVector["dx"] = np.random.uniform(-gazeForce, gazeForce)
            self.currentGazeVector["dy"] = np.random.uniform(-gazeForce, gazeForce)
        
        return landmarks
    
    def _updateAnomalyState(self):
        """Atualiza estado de anomalias baseado em probabilidades"""
        
        currentTime = self.currentTimestamp
        
        # Se já há uma anomalia ativa, verificar se deve terminar
        if self.currentAnomalyType != CameraAnomalyType.NORMAL:
            if currentTime - self.anomalyStartTime >= self.anomalyDuration:
                self.currentAnomalyType = CameraAnomalyType.NORMAL
                self.logger.debug(f"Camera anomaly ended at {currentTime:.3f}s")
            return
        
        # Verificar se deve injetar nova anomalia
        if not self.anomalyConfig["enabled"]:
            return
        
        # Intervalo mínimo entre anomalias
        if currentTime - self.lastAnomalyTime < self.anomalyConfig["minInterval"]:
            return
        
        # Probabilidade de anomalia
        if np.random.random() < self.anomalyChance:
            # Escolher tipo de anomalia baseado nas probabilidades configuradas
            anomalyTypes = []
            weights = []
            
            for anomalyName, config in self.anomalyTypes.items():
                try:
                    anomalyType = CameraAnomalyType(anomalyName)
                    if anomalyType != CameraAnomalyType.NORMAL:
                        anomalyTypes.append(anomalyType)
                        weights.append(config["probability"])
                except ValueError:
                    continue
            
            if anomalyTypes and weights:
                # Normalizar pesos
                totalWeight = sum(weights)
                normalizedWeights = [w / totalWeight for w in weights]
                
                self.currentAnomalyType = np.random.choice(anomalyTypes, p=normalizedWeights)
                
                self.anomalyStartTime = currentTime
                self.lastAnomalyTime = currentTime
                
                # Duração baseada na configuração
                anomalyConfig = self.anomalyTypes[self.currentAnomalyType.value.lower()]
                durationRange = anomalyConfig["durationRange"]
                self.anomalyDuration = np.random.uniform(durationRange[0], durationRange[1])
                
                self.logger.warning(f"Camera anomaly started: {self.currentAnomalyType.value} for {self.anomalyDuration:.1f}s")
    
    def forceAnomaly(self, anomalyType: str, duration: float = 10.0):
        """
        Força injeção de anomalia específica.
        
        Args:
            anomalyType: Tipo de anomalia ("low_blink_rate", "gaze_drift", etc.)
            duration: Duração da anomalia em segundos
        """
        
        try:
            self.currentAnomalyType = CameraAnomalyType(anomalyType)
            self.anomalyStartTime = self.currentTimestamp
            self.lastAnomalyTime = self.currentTimestamp
            self.anomalyDuration = duration
            
            self.logger.warning(f"Forced camera anomaly: {anomalyType} for {duration}s")
            
        except ValueError:
            self.logger.error(f"Unknown camera anomaly type: {anomalyType}")
    
    def forceAttentionPattern(self, pattern: str, duration: float = 15.0):
        """
        Força padrão de atenção específico.
        
        Args:
            pattern: Padrão de atenção ("focused", "drowsy", etc.)
            duration: Duração do padrão em segundos
        """
        
        try:
            self.currentAttentionPattern = AttentionPattern(pattern)
            self.patternStartTime = self.currentTimestamp
            self.patternDuration = duration
            
            self.logger.info(f"Forced attention pattern: {pattern} for {duration}s")
            
        except ValueError:
            self.logger.error(f"Unknown attention pattern: {pattern}")
    
    def getStatus(self) -> Dict[str, Any]:
        """
        Retorna status atual do gerador.
        
        Returns:
            Status detalhado do gerador
        """
        
        return {
            "generatorType": "CameraFaceLandmarks",
            "fps": self.fps,
            "landmarksCount": self.landmarksCount,
            "currentTimestamp": self.currentTimestamp,
            "frameCounter": self.frameCounter,
            "currentAttentionPattern": self.currentAttentionPattern.value,
            "currentAnomalyType": self.currentAnomalyType.value,
            "anomalyActive": self.currentAnomalyType != CameraAnomalyType.NORMAL,
            "anomalyTimeRemaining": max(0, (self.anomalyStartTime + self.anomalyDuration) - self.currentTimestamp),
            "patternTimeRemaining": max(0, (self.patternStartTime + self.patternDuration) - self.currentTimestamp),
            "blinkState": {
                "isBlinking": self.isBlinking,
                "blinkCounter": self.blinkCounter,
                "currentEar": self.currentEar,
                "recentBlinkRate": self._calculateBlinkRate()
            },
            "gazeState": {
                "currentGaze": self.currentGazeVector.copy(),
                "gazeTarget": self.gazeTarget.copy()
            },
            "detectionQuality": {
                "imageSize": self.imageSize
            },
            "config": {
                "anomalyChance": self.anomalyChance,
                "blinkRange": self.blinkConfig["normalRange"],
                "earRange": self.earConfig["normalRange"]
            }
        }
    
    def reset(self):
        """Reset do estado interno do gerador"""
        
        self.currentTimestamp = 0.0
        self.frameCounter = 0
        self.lastAnomalyTime = 0.0
        self.currentAnomalyType = CameraAnomalyType.NORMAL
        self.anomalyDuration = 0.0
        self.anomalyStartTime = 0.0
        self.currentAttentionPattern = AttentionPattern.FOCUSED
        self.patternStartTime = 0.0
        self.patternDuration = 0.0
        
        # Reset estado facial
        self.currentEar = 0.3
        self.isBlinking = False
        self.blinkStartTime = 0.0
        self.lastBlinkTime = 0.0
        self.blinkCounter = 0
        self.recentBlinkTimes.clear()
        
        # Reset gaze
        self.currentGazeVector = {"dx": 0.0, "dy": 0.0}
        self.gazeTarget = {"dx": 0.0, "dy": 0.0}
        
        # Reset posição
        self.headPosition = np.array(self.anatomyPositions["faceCenter"])
        self.headRotation = np.array([0.0, 0.0, 0.0])
        self.microMovementPhase = 0.0
        
        # Regenerar landmarks base
        self.baseLandmarks = self._generateBaseFaceLandmarks()
        self.currentLandmarks = self.baseLandmarks.copy()
        
        self.logger.info("CameraFaceLandmarksGenerator reset completed")

# Instância global
cameraFaceLandmarksGenerator = CameraFaceLandmarksGenerator()