import os
import uuid
from pathlib import Path
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
from deepface import DeepFace

# --- Runtime & Configuration Setup ---
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')

ATTACKER_MODELS = {
    'Facenet512': (160, 160),
    'ArcFace': (112, 112),
    'GhostFaceNet': (112, 112),
    'VGG-Face': (224, 224),
}

VICTIM_MODELS = ['Facenet512', 'ArcFace', 'GhostFaceNet', 'VGG-Face', 'IR152']

ALL_ATTACKS = [
    'PGD',
    'MI_FGSM',
    'TI_FGSM',
    'SI_NI_FGSM',
    'SEMANTIC_MI_FGSM',
    'MI_ADMIX_DI_TI',
    'DYNAMIC_MORPH'
]

ATTACK_COLS = {
    'PGD': 'pgd_path',
    'MI_FGSM': 'mi_fgsm_path',
    'TI_FGSM': 'ti_fgsm_path',
    'SI_NI_FGSM': 'si_ni_fgsm_path',
    'MI_ADMIX_DI_TI': 'mi_admix_di_ti_path',
    'DYNAMIC_MORPH': 'dynamic_morph_path'
}

EPSILON = 0.062
NUM_ITER = 5
DECAY = 1.0

# --- Helper Utilities ---
def configure_cpu_runtime(tf_threads: int = 1) -> None:
    try:
        tf.config.set_visible_devices([], 'GPU')
    except Exception:
        pass
    try:
        tf.config.threading.set_intra_op_parallelism_threads(tf_threads)
        tf.config.threading.set_inter_op_parallelism_threads(tf_threads)
    except Exception:
        pass


def resolve_image_path(path: str, dataset_root: str) -> str:
    value = str(path)
    if os.path.exists(value):
        return value
    marker = 'dataset_extractedfaces/'
    if marker in value:
        rel = value.split(marker, 1)[1]
        candidate = os.path.join(dataset_root, rel)
        if os.path.exists(candidate):
            return candidate
    return os.path.join(dataset_root, value.lstrip('/'))


def load_and_preprocess(path: str, input_size):
    img = Image.open(path).convert('RGB').resize(input_size)
    arr = np.array(img).astype('float32') / 255.0
    return (arr - 0.5) * 2.0


def denormalize(x: np.ndarray) -> np.ndarray:
    x = (x + 1.0) / 2.0
    return np.clip(x * 255, 0, 255).astype(np.uint8)


def compute_embedding(model, x):
    out = model(x, training=False)
    if isinstance(out, (tuple, list)):
        out = out[0]
    return tf.nn.l2_normalize(out, axis=1)


def attack_loss(cos, attack_type: str):
    return tf.reduce_mean(cos if str(attack_type).strip().lower() == 'impersonation_attack' else (1 - cos))


def save_adv(img_uint8: np.ndarray, attack_name: str, src: str, tgt: str, attack_type: str, model_name: str, row_id: int, adv_root: str) -> str:
    out_dir = Path(adv_root) / model_name / attack_name
    out_dir.mkdir(parents=True, exist_ok=True)
    s = Path(src).stem.replace(' ', '_')
    t = Path(tgt).stem.replace(' ', '_')
    rand = uuid.uuid4().hex[:8]
    name = f'adv_r{row_id}_{s}_to_{t}_{attack_type}_{rand}.png'
    path = out_dir / name
    Image.fromarray(img_uint8).save(path)
    return str(path.resolve())


def gaussian_kernel(k=15, sigma=3.0, ch=3):
    x = tf.range(-k // 2 + 1, k // 2 + 1, dtype=tf.float32)
    g = tf.exp(-tf.square(x) / (2 * sigma**2))
    g /= tf.reduce_sum(g)
    kernel = tf.tensordot(g, g, axes=0)
    kernel = kernel[:, :, None, None]
    return tf.tile(kernel, [1, 1, ch, 1])


def input_diversity(x, input_size, prob=0.7):
    if tf.random.uniform([]) > prob:
        return x
    img_size = input_size[0]
    rnd = tf.random.uniform([], int(0.9 * img_size), img_size, dtype=tf.int32)
    x_resized = tf.image.resize(x, (rnd, rnd))
    pad_total = img_size - rnd
    pad_top = tf.random.uniform([], 0, pad_total + 1, dtype=tf.int32)
    pad_bottom = pad_total - pad_top
    pad_left = tf.random.uniform([], 0, pad_total + 1, dtype=tf.int32)
    pad_right = pad_total - pad_left
    x_padded = tf.pad(x_resized, [[0, 0], [pad_top, pad_bottom], [pad_left, pad_right], [0, 0]])
    return tf.image.resize(x_padded, input_size)


# --- Baseline Attack Implementations ---
def pgd_attack(model, x, tgt_emb, attack_type, random_start=True):
    if random_start:
        noise = tf.random.uniform(tf.shape(x), minval=-EPSILON, maxval=EPSILON, dtype=x.dtype)
        adv = tf.clip_by_value(x + noise, -1.0, 1.0)
    else:
        adv = tf.identity(x)
    alpha = EPSILON / NUM_ITER
    tgt_emb = tf.nn.l2_normalize(tgt_emb, axis=1)
    for _ in range(NUM_ITER):
        with tf.GradientTape() as tape:
            tape.watch(adv)
            emb = compute_embedding(model, adv)
            cos = tf.reduce_sum(emb * tgt_emb, axis=1)
            loss = attack_loss(cos, attack_type)
        grad = tape.gradient(loss, adv)
        adv = adv + alpha * tf.sign(grad)
        adv = tf.clip_by_value(adv, x - EPSILON, x + EPSILON)
        adv = tf.clip_by_value(adv, -1.0, 1.0)
    return adv


def mi_fgsm(model, x, tgt_emb, attack_type):
    adv = tf.identity(x)
    g = tf.zeros_like(x)
    alpha = EPSILON / NUM_ITER
    tgt_emb = tf.nn.l2_normalize(tgt_emb, axis=1)
    for _ in range(NUM_ITER):
        with tf.GradientTape() as tape:
            tape.watch(adv)
            emb = compute_embedding(model, adv)
            cos = tf.reduce_sum(emb * tgt_emb, axis=1)
            loss = attack_loss(cos, attack_type)
        grad = tape.gradient(loss, adv)
        grad = grad / (tf.reduce_mean(tf.abs(grad)) + 1e-8)
        g = DECAY * g + grad
        adv = adv + alpha * tf.sign(g)
        adv = tf.clip_by_value(adv, x - EPSILON, x + EPSILON)
        adv = tf.clip_by_value(adv, -1.0, 1.0)
    return adv


def ti_fgsm(model, x, tgt_emb, attack_type):
    adv = tf.identity(x)
    alpha = EPSILON / NUM_ITER
    kernel = gaussian_kernel()
    tgt_emb = tf.nn.l2_normalize(tgt_emb, axis=1)
    for _ in range(NUM_ITER):
        with tf.GradientTape() as tape:
            tape.watch(adv)
            emb = compute_embedding(model, adv)
            cos = tf.reduce_sum(emb * tgt_emb, axis=1)
            loss = attack_loss(cos, attack_type)
        grad = tape.gradient(loss, adv)
        grad = tf.nn.depthwise_conv2d(grad, kernel, [1, 1, 1, 1], 'SAME')
        adv = adv + alpha * tf.sign(grad)
        adv = tf.clip_by_value(adv, x - EPSILON, x + EPSILON)
        adv = tf.clip_by_value(adv, -1.0, 1.0)
    return adv


def si_ni_fgsm(model, x, tgt_emb, attack_type):
    adv = tf.identity(x)
    g = tf.zeros_like(x)
    alpha = EPSILON / NUM_ITER
    tgt_emb = tf.nn.l2_normalize(tgt_emb, axis=1)
    scales = (1.0, 0.5, 0.25, 0.125, 0.0625)
    for _ in range(NUM_ITER):
        nes = adv + DECAY * alpha * g
        grad_sum = tf.zeros_like(x)
        for s in scales:
            with tf.GradientTape() as tape:
                tape.watch(nes)
                emb = compute_embedding(model, nes * s)
                cos = tf.reduce_sum(emb * tgt_emb, axis=1)
                loss = attack_loss(cos, attack_type)
            grad_sum += tape.gradient(loss, nes)
        grad = grad_sum / len(scales)
        grad = grad / (tf.reduce_mean(tf.abs(grad)) + 1e-8)
        g = DECAY * g + grad
        adv = adv + alpha * tf.sign(g)
        adv = tf.clip_by_value(adv, x - EPSILON, x + EPSILON)
        adv = tf.clip_by_value(adv, -1.0, 1.0)
    return adv


def mi_admix_di_ti(model, x, tgt_emb, attack_type, pool_imgs, input_size):
    adv = tf.identity(x)
    g = tf.zeros_like(x)
    alpha = EPSILON / NUM_ITER
    tgt_emb = tf.nn.l2_normalize(tgt_emb, axis=1)
    kernel = gaussian_kernel()
    n_pool = tf.shape(pool_imgs)[0]
    for _ in range(NUM_ITER):
        with tf.GradientTape() as tape:
            tape.watch(adv)
            idx = tf.random.uniform([3], 0, n_pool, dtype=tf.int32)
            others = tf.gather(pool_imgs, idx)
            adv_rep = tf.repeat(adv, 3, axis=0)
            mixed = adv_rep + 0.2 * (others - adv_rep)
            batch = input_diversity(mixed, input_size)
            emb = compute_embedding(model, batch)
            tgt_rep = tf.repeat(tgt_emb, 3, axis=0)
            cos = tf.reduce_sum(emb * tgt_rep, axis=1)
            loss = attack_loss(cos, attack_type)
        grad = tape.gradient(loss, adv)
        grad = tf.nn.depthwise_conv2d(grad, kernel, [1, 1, 1, 1], 'SAME')
        grad = grad / (tf.reduce_mean(tf.abs(grad)) + 1e-8)
        g = DECAY * g + grad
        adv = adv + alpha * tf.sign(g)
        adv = tf.clip_by_value(adv, x - EPSILON, x + EPSILON)
        adv = tf.clip_by_value(adv, -1.0, 1.0)
    return adv


# --- Custom Research Attack Implementations (Assignment 4) ---
def get_semantic_mask(input_size):
    """Generates a blurred spatial mask targeting the eyes and nose."""
    h, w = input_size
    mask = np.zeros((h, w, 3), dtype=np.float32)
    
    y1, y2 = int(h * 0.35), int(h * 0.55)
    xl1, xl2 = int(w * 0.20), int(w * 0.45)
    xr1, xr2 = int(w * 0.55), int(w * 0.80)
    yn1, yn2 = int(h * 0.50), int(h * 0.70)
    xn1, xn2 = int(w * 0.40), int(w * 0.60)
    
    mask[y1:y2, xl1:xl2, :] = 1.0  
    mask[y1:y2, xr1:xr2, :] = 1.0  
    mask[yn1:yn2, xn1:xn2, :] = 1.0  
    
    mask = cv2.GaussianBlur(mask, (35, 35), 0)
    return tf.convert_to_tensor(mask, dtype=tf.float32)


def semantic_mi_fgsm(model, x, tgt_emb, attack_type, input_size):
    """Semantic Feature-Level Transfer Attack Variant"""
    adv = tf.identity(x)
    g = tf.zeros_like(x)
    alpha = EPSILON / NUM_ITER
    tgt_emb = tf.nn.l2_normalize(tgt_emb, axis=1)
    
    feature_mask = get_semantic_mask(input_size)
    
    for _ in range(NUM_ITER):
        with tf.GradientTape() as tape:
            tape.watch(adv)
            emb = compute_embedding(model, adv)
            cos = tf.reduce_sum(emb * tgt_emb, axis=1)
            loss = attack_loss(cos, attack_type)
            
        grad = tape.gradient(loss, adv)
        grad = grad * feature_mask
        grad = grad / (tf.reduce_mean(tf.abs(grad)) + 1e-8)
        g = DECAY * g + grad
        adv = adv + alpha * tf.sign(g)
        adv = tf.clip_by_value(adv, x - EPSILON, x + EPSILON)
        adv = tf.clip_by_value(adv, -1.0, 1.0)
        
    return adv


def dynamic_morph_mi_fgsm(model, src, tgt, attack_type, input_size):
    """D-FMA (Pre-aligned Semantic Mixing Attack)"""
    h, w = input_size
    mask = np.zeros((h, w, 3), dtype=np.float32)
    
    mask[int(h*0.35):int(h*0.52), int(w*0.20):int(w*0.45), :] = 1.0  
    mask[int(h*0.35):int(h*0.52), int(w*0.55):int(w*0.80), :] = 1.0  
    mask[int(h*0.48):int(h*0.70), int(w*0.38):int(w*0.62), :] = 1.0  
    
    mask = cv2.GaussianBlur(mask, (25, 25), 0)
    tf_mask = tf.convert_to_tensor(mask, dtype=tf.float32)
    
    morphed_tensor = (tgt * tf_mask) + (src * (1.0 - tf_mask))
    
    adv = tf.identity(morphed_tensor)
    g = tf.zeros_like(adv)
    alpha = EPSILON / NUM_ITER
    tgt_emb = tf.nn.l2_normalize(compute_embedding(model, tgt), axis=1)
    
    for _ in range(NUM_ITER):
        with tf.GradientTape() as tape:
            tape.watch(adv)
            emb = compute_embedding(model, adv)
            cos = tf.reduce_sum(emb * tgt_emb, axis=1)
            loss = attack_loss(cos, attack_type)
            
        grad = tape.gradient(loss, adv)
        grad = grad / (tf.reduce_mean(tf.abs(grad)) + 1e-8)
        g = DECAY * g + grad
        adv = adv + alpha * tf.sign(g)
        adv = tf.clip_by_value(adv, morphed_tensor - EPSILON, morphed_tensor + EPSILON)
        adv = tf.clip_by_value(adv, -1.0, 1.0)
        
    return adv


# --- Execution Router ---
def build_attacker(model_name: str):
    return DeepFace.build_model(model_name).model


def run_attack(attack_name: str, model, src, tgt, attack_type: str, input_size):
    tgt_emb = compute_embedding(model, tgt)
    if attack_name == 'PGD':
        return pgd_attack(model, src, tgt_emb, attack_type)
    if attack_name == 'MI_FGSM':
        return mi_fgsm(model, src, tgt_emb, attack_type)
    if attack_name == 'TI_FGSM':
        return ti_fgsm(model, src, tgt_emb, attack_type)
    if attack_name == 'SI_NI_FGSM':
        return si_ni_fgsm(model, src, tgt_emb, attack_type)
    if attack_name == 'DYNAMIC_MORPH':
        return dynamic_morph_mi_fgsm(model, src, tgt, attack_type, input_size)
    if attack_name == 'SEMANTIC_MI_FGSM':
        return semantic_mi_fgsm(model, src, tgt_emb, attack_type, input_size)
    if attack_name == 'MI_ADMIX_DI_TI':
        pool_imgs = tf.concat([src, tgt, src], axis=0)
        return mi_admix_di_ti(model, src, tgt_emb, attack_type, pool_imgs, input_size)
    raise ValueError(f'Unsupported attack: {attack_name}')