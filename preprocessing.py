"""
=============================================================
  Tiền xử lý Dataset nhãn sản phẩm
  Tên file format: {category}.{product_idx}.{face_idx}.jpg

  Ví dụ:
    meat.2.1.jpg          → category=meat,   product=2, face=1
    snack.1.2.jpg         → category=snack,  product=1, face=2
    fermented_food.1.1.jpg→ category=fermented_food, product=1, face=1

  Quy tắc split: chia theo PRODUCT (không phải ảnh)
    → meat.2.1 và meat.2.2 (cùng sản phẩm) luôn ở cùng 1 tập
    → tránh data leakage giữa train/val/test

Cấu trúc INPUT:
  new_dataset/
  ├── meat.1.1.jpg
  ├── meat.1.2.jpg
  ├── snack.1.1.jpg
  ├── fermented_food.2.1.jpg
  └── ...

Cấu trúc OUTPUT:
  processed/
  ├── images/
  │   ├── train/
  │   ├── val/
  │   └── test/
  ├── labels/
  │   ├── train/
  │   ├── val/
  │   └── test/
  ├── metadata.csv       ← thống kê toàn bộ ảnh đã parse
  └── data.yaml
=============================================================
"""

import re
import cv2
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from sklearn.model_selection import train_test_split
from tqdm import tqdm

import albumentations as A

# ─────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
# 1. CẤU HÌNH
# ═══════════════════════════════════════════════════════════════
class Config:

    # ── Đường dẫn ──────────────────────────────────────────────
    INPUT_DIR  = "new_dataset"   # Thư mục chứa toàn bộ ảnh gốc
    LABEL_DIR  = "labels"        # YOLO .txt (nếu có, cùng tên với ảnh)
    OUTPUT_DIR = "processed"

    # ── Kích thước đầu ra ──────────────────────────────────────
    TARGET_SIZE = 640            # YOLO26 / YOLOv8 chuẩn

    # ── Split theo product (tránh data leakage) ────────────────
    TRAIN_RATIO = 0.70
    VAL_RATIO   = 0.15
    TEST_RATIO  = 0.15
    RANDOM_SEED = 42

    # ── Regex parse tên file ───────────────────────────────────
    # Khớp: category.product_idx.face_idx.ext
    # Ví dụ: fermented_food.2.1.jpg | meat.10.2.png
    FILENAME_RE = re.compile(
        r"^(?P<category>[a-zA-Z][a-zA-Z_]*)
         r"\.(?P<product_idx>\d+)"
         r"\.(?P<face_idx>\d+)"
         r"\.[a-zA-Z]{3,4}$"
    )

    # ── Lọc chất lượng ảnh ─────────────────────────────────────
    MIN_IMAGE_SIZE   = 80        # px — bỏ ảnh nhỏ hơn
    MAX_ASPECT_RATIO = 8.0       # bỏ ảnh tỉ lệ quá dài/rộng
    JPEG_QUALITY     = 95

    # ── Augmentation cho tập train ─────────────────────────────
    AUGMENT_TRAIN  = True
    AUGMENT_COPIES = 2           # số bản aug thêm mỗi ảnh train

    # ── Cân bằng category ──────────────────────────────────────
    BALANCE        = True
    BALANCE_TARGET = 80          # số product tối thiểu mỗi category

    # ── Xử lý song song ────────────────────────────────────────
    NUM_WORKERS = 4


cfg = Config()


# ═══════════════════════════════════════════════════════════════
# 2. PARSE TÊN FILE
# ═══════════════════════════════════════════════════════════════
def parse_filename(path: Path) -> dict | None:
    """
    Parse tên file theo format category.product_idx.face_idx.ext

    Trả về dict với các key:
        category, product_idx, face_idx, product_key, path
    Trả về None nếu không khớp format (ảnh hash, tên lạ, v.v.)
    """
    m = cfg.FILENAME_RE.match(path.name)
    if not m:
        return None

    category    = m.group("category")
    product_idx = int(m.group("product_idx"))
    face_idx    = int(m.group("face_idx"))

    return {
        "filename":    path.name,
        "path":        path,
        "category":    category,
        "product_idx": product_idx,
        "face_idx":    face_idx,
        # Key duy nhất cho mỗi sản phẩm (dùng để group các mặt)
        "product_key": f"{category}.{product_idx}",
    }


def load_dataset(input_dir: str) -> pd.DataFrame:
    """
    Quét toàn bộ thư mục, parse tên file hợp lệ.
    Bỏ qua ảnh hash hoặc không đúng format.
    """
    input_dir = Path(input_dir)
    exts      = {".jpg", ".jpeg", ".png", ".webp"}
    all_files = [p for p in input_dir.iterdir()
                 if p.suffix.lower() in exts]

    log.info(f"Tổng file tìm thấy: {len(all_files)}")

    records   = []
    skipped   = []
    for p in all_files:
        info = parse_filename(p)
        if info:
            records.append(info)
        else:
            skipped.append(p.name)

    log.info(f"  ✅ Hợp lệ (đúng format): {len(records)}")
    log.info(f"  ⏭️  Bỏ qua (hash/sai format): {len(skipped)}")
    if skipped:
        log.debug(f"  Ví dụ bị bỏ: {skipped[:3]}")

    df = pd.DataFrame(records)
    df = df.sort_values(["category", "product_idx", "face_idx"]).reset_index(drop=True)
    return df


# ═══════════════════════════════════════════════════════════════
# 3. THỐNG KÊ & VISUALIZE
# ═══════════════════════════════════════════════════════════════
def print_stats(df: pd.DataFrame):
    """In thống kê dataset ra console."""
    print("\n" + "═" * 58)
    print("  THỐNG KÊ DATASET")
    print("═" * 58)
    print(f"  Tổng ảnh hợp lệ  : {len(df)}")
    print(f"  Số category      : {df['category'].nunique()}")
    print(f"  Số sản phẩm      : {df['product_key'].nunique()}")
    print(f"  Số mặt tối đa    : {df['face_idx'].max()}")

    print("\n  Phân phối theo category:")
    cat_stats = df.groupby("category").agg(
        ảnh=("filename", "count"),
        sản_phẩm=("product_key", "nunique"),
        mặt_tb=("face_idx", "mean")
    ).round(1)
    for cat, row in cat_stats.iterrows():
        bar = "█" * int(row["ảnh"] // 5)
        print(f"    {cat:<20}: {int(row['ảnh']):>4} ảnh | "
              f"{int(row['sản_phẩm']):>3} SP | "
              f"~{row['mặt_tb']:.1f} mặt/SP  {bar}")

    print("\n  Phân phối số mặt mỗi sản phẩm:")
    face_dist = df.groupby("product_key")["face_idx"].count().value_counts().sort_index()
    for n_faces, count in face_dist.items():
        print(f"    {n_faces} mặt : {count} sản phẩm")
    print("═" * 58 + "\n")


def plot_stats(df: pd.DataFrame, save_path: str = "dataset_stats.png"):
    """Vẽ biểu đồ phân phối category và số mặt."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colors = ["#2ecc71","#3498db","#e74c3c","#f39c12","#9b59b6","#1abc9c","#e67e22"]

    # Biểu đồ 1: Số ảnh theo category
    cat_count = df["category"].value_counts()
    bars = axes[0].bar(cat_count.index, cat_count.values,
                       color=colors[:len(cat_count)], edgecolor="white")
    for bar, v in zip(bars, cat_count.values):
        axes[0].text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + 0.5, str(v),
                     ha="center", fontsize=9, fontweight="bold")
    axes[0].set_title("Số ảnh theo Category", fontweight="bold")
    axes[0].set_xlabel("Category")
    axes[0].set_ylabel("Số ảnh")
    axes[0].tick_params(axis="x", rotation=30)

    # Biểu đồ 2: Phân phối số mặt mỗi sản phẩm
    face_dist = df.groupby("product_key")["face_idx"].count().value_counts().sort_index()
    axes[1].bar(face_dist.index.astype(str), face_dist.values,
                color="#3498db", edgecolor="white")
    axes[1].set_title("Số mặt mỗi Sản phẩm", fontweight="bold")
    axes[1].set_xlabel("Số mặt")
    axes[1].set_ylabel("Số sản phẩm")

    plt.suptitle("Thống kê Dataset", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    log.info(f"Đã lưu biểu đồ: {save_path}")
    plt.show()


# ═══════════════════════════════════════════════════════════════
# 4. SPLIT THEO PRODUCT (tránh data leakage)
# ═══════════════════════════════════════════════════════════════
def split_by_product(df: pd.DataFrame) -> pd.DataFrame:
    """
    Chia train/val/test theo product_key (không phải theo ảnh).

    Tại sao quan trọng:
        meat.2.1.jpg và meat.2.2.jpg là 2 mặt của CÙNG 1 sản phẩm.
        Nếu chia theo ảnh, mặt 1 có thể vào train, mặt 2 vào val
        → model học được sản phẩm đó từ train → kết quả val bị inflate.
        → Chia theo product đảm bảo sản phẩm chưa thấy ở val/test.
    """
    # Lấy danh sách product duy nhất, kèm category để stratify
    products = (
        df[["product_key", "category"]]
        .drop_duplicates("product_key")
        .reset_index(drop=True)
    )

    labels = products["category"].tolist()
    can_stratify = len(set(labels)) > 1

    # Split products → train / (val + test)
    train_prod, temp_prod = train_test_split(
        products,
        test_size=(cfg.VAL_RATIO + cfg.TEST_RATIO),
        random_state=cfg.RANDOM_SEED,
        stratify=labels if can_stratify else None
    )

    # Split (val + test) → val / test
    val_ratio_adj = cfg.VAL_RATIO / (cfg.VAL_RATIO + cfg.TEST_RATIO)
    temp_labels   = temp_prod["category"].tolist()

    val_prod, test_prod = train_test_split(
        temp_prod,
        test_size=(1 - val_ratio_adj),
        random_state=cfg.RANDOM_SEED,
        stratify=temp_labels if len(set(temp_labels)) > 1 else None
    )

    # Gán nhãn split vào df gốc (mỗi ảnh thừa kế split từ product)
    split_map = {}
    for pk in train_prod["product_key"]: split_map[pk] = "train"
    for pk in val_prod["product_key"]:   split_map[pk] = "val"
    for pk in test_prod["product_key"]:  split_map[pk] = "test"

    df = df.copy()
    df["split"] = df["product_key"].map(split_map)

    # Log kết quả
    for s in ["train", "val", "test"]:
        sub = df[df["split"] == s]
        n_prod = sub["product_key"].nunique()
        n_img  = len(sub)
        log.info(f"  {s:<6}: {n_prod:>4} sản phẩm | {n_img:>4} ảnh")

    return df


# ═══════════════════════════════════════════════════════════════
# 5. OVERSAMPLE CATEGORY ÍT (tuỳ chọn)
# ═══════════════════════════════════════════════════════════════
def oversample_train(df: pd.DataFrame) -> pd.DataFrame:
    """
    Duplicate các product thuộc category ít ảnh trong tập train.
    Augmentation sẽ tạo ra bản ảnh thực sự khác nhau khi xử lý.
    """
    train_df = df[df["split"] == "train"].copy()
    other_df = df[df["split"] != "train"].copy()

    # Đếm số product mỗi category trong train
    cat_prod_count = (
        train_df.groupby("category")["product_key"]
        .nunique()
    )
    target = cfg.BALANCE_TARGET
    frames = [train_df]

    for cat, n_prod in cat_prod_count.items():
        if n_prod < target:
            needed   = target - n_prod
            cat_rows = train_df[train_df["category"] == cat]
            extra    = cat_rows.sample(
                n=needed * cat_rows["face_idx"].nunique(),
                replace=True,
                random_state=cfg.RANDOM_SEED
            )
            frames.append(extra)
            log.info(f"  Oversample '{cat}': +{needed} product tương đương")

    new_train = pd.concat(frames, ignore_index=True).sample(
        frac=1, random_state=cfg.RANDOM_SEED
    )
    return pd.concat([new_train, other_df], ignore_index=True)


# ═══════════════════════════════════════════════════════════════
# 6. LÀM SẠCH ẢNH
# ═══════════════════════════════════════════════════════════════
class ImageCleaner:
    """
    Xử lý đặc thù ảnh bao bì thực phẩm chụp tay:
    - Phản chiếu sáng trên bao bì bóng → CLAHE
    - Màu lệch do đèn siêu thị → white balance
    """

    @staticmethod
    def remove_glare(img: np.ndarray) -> np.ndarray:
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)

    @staticmethod
    def white_balance(img: np.ndarray) -> np.ndarray:
        img_f = img.astype(np.float32)
        avg   = img_f.mean(axis=(0, 1))
        gray  = avg.mean()
        scale = np.clip(gray / (avg + 1e-6), 0.5, 2.0)
        return np.clip(img_f * scale, 0, 255).astype(np.uint8)

    def clean(self, img: np.ndarray) -> np.ndarray:
        img = self.remove_glare(img)
        img = self.white_balance(img)
        return img


# ═══════════════════════════════════════════════════════════════
# 7. LETTERBOX RESIZE + ADJUST LABELS
# ═══════════════════════════════════════════════════════════════
def letterbox(img: np.ndarray, size: int = 640) -> tuple[np.ndarray, dict]:
    """Resize về size×size giữ aspect ratio, padding màu xám."""
    h, w  = img.shape[:2]
    scale = min(size / w, size / h)
    nw, nh = int(w * scale), int(h * scale)
    img_r  = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)

    pt = (size - nh) // 2;  pb = size - nh - pt
    pl = (size - nw) // 2;  pr = size - nw - pl

    img_pad = cv2.copyMakeBorder(
        img_r, pt, pb, pl, pr,
        cv2.BORDER_CONSTANT, value=(114, 114, 114)
    )
    return img_pad, {"orig_w": w, "orig_h": h,
                     "scale": scale, "pad_top": pt, "pad_left": pl}


def adjust_labels(label_path: Path, meta: dict, size: int = 640) -> list[str]:
    """Điều chỉnh YOLO bbox sau letterbox. Trả về [] nếu không có label."""
    if not label_path.exists():
        return []
    ow, oh = meta["orig_w"], meta["orig_h"]
    sc     = meta["scale"]
    pl, pt = meta["pad_left"], meta["pad_top"]
    out    = []
    with open(label_path) as f:
        for line in f:
            p = line.strip().split()
            if len(p) < 5:
                continue
            cx = np.clip((float(p[1]) * ow * sc + pl) / size, 0, 1)
            cy = np.clip((float(p[2]) * oh * sc + pt) / size, 0, 1)
            bw = np.clip((float(p[3]) * ow * sc)       / size, 0, 1)
            bh = np.clip((float(p[4]) * oh * sc)       / size, 0, 1)
            out.append(f"{p[0]} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
    return out


# ═══════════════════════════════════════════════════════════════
# 8. AUGMENTATION
# ═══════════════════════════════════════════════════════════════
def build_augmentation() -> A.Compose:
    """
    Augmentation phù hợp ảnh bao bì sản phẩm:
    - Flip ngang: nhãn thường đối xứng
    - Xoay nhẹ: ảnh chụp tay bị nghiêng
    - Thay đổi sáng/tương phản: điều kiện ánh sáng siêu thị
    - Blur nhẹ: mô phỏng ảnh chụp chuyển động
    """
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(
            shift_limit=0.05, scale_limit=0.2, rotate_limit=15,
            border_mode=cv2.BORDER_CONSTANT, value=(114, 114, 114), p=0.6
        ),
        A.OneOf([
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=1.0),
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=30, val_shift_limit=20, p=1.0),
        ], p=0.7),
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 5), p=1.0),
            A.MotionBlur(blur_limit=5, p=1.0),
        ], p=0.3),
        A.GaussNoise(var_limit=(5.0, 25.0), p=0.2),
        A.CoarseDropout(
            num_holes_range=(1, 3),
            hole_height_range=(10, 30), hole_width_range=(10, 30),
            fill_value=(114, 114, 114), p=0.15
        ),
    ],
    bbox_params=A.BboxParams(
        format="yolo", label_fields=["class_labels"],
        min_visibility=0.3, clip=True
    ))


# ═══════════════════════════════════════════════════════════════
# 9. XỬ LÝ MỖI ẢNH
# ═══════════════════════════════════════════════════════════════
class SampleProcessor:

    def __init__(self, output_dir: Path):
        self.out     = output_dir
        self.cleaner = ImageCleaner()
        self.aug     = build_augmentation()

    def process(self, row: dict) -> bool:
        """
        Xử lý 1 ảnh:
            load → validate → clean → letterbox
            → adjust_labels → save [→ augment nếu train]
        """
        img_path = Path(row["path"])
        split    = row["split"]
        is_train = (split == "train")

        try:
            img = cv2.imread(str(img_path))
            if img is None:
                return False

            h, w = img.shape[:2]
            if min(h, w) < cfg.MIN_IMAGE_SIZE:
                return False
            if max(h, w) / min(h, w) > cfg.MAX_ASPECT_RATIO:
                return False

            # Clean
            img = self.cleaner.clean(img)

            # Letterbox resize
            img_lb, meta = letterbox(img, cfg.TARGET_SIZE)

            # Adjust labels
            label_path = Path(cfg.LABEL_DIR) / img_path.with_suffix(".txt").name
            adj_labels = adjust_labels(label_path, meta, cfg.TARGET_SIZE)

            # Augmentation (train only)
            if is_train and cfg.AUGMENT_TRAIN:
                bboxes, cls_ids = self._parse_yolo(adj_labels)
                if bboxes:
                    for i in range(cfg.AUGMENT_COPIES):
                        try:
                            res = self.aug(
                                image=img_lb,
                                bboxes=bboxes,
                                class_labels=cls_ids
                            )
                            stem = f"{img_path.stem}_aug{i}"
                            self._save(res["image"],
                                       res["bboxes"], res["class_labels"],
                                       stem, split)
                        except Exception:
                            pass

            # Lưu ảnh gốc (đã clean + resize)
            self._save_raw(img_lb, adj_labels, img_path.stem, split)
            return True

        except Exception as e:
            log.debug(f"Lỗi {img_path.name}: {e}")
            return False

    def _parse_yolo(self, lines):
        bboxes, cls_ids = [], []
        for line in lines:
            p = line.split()
            if len(p) == 5:
                cls_ids.append(int(p[0]))
                bboxes.append([float(x) for x in p[1:]])
        return bboxes, cls_ids

    def _save(self, img, bboxes, cls_ids, stem, split):
        lines = [f"{c} {b[0]:.6f} {b[1]:.6f} {b[2]:.6f} {b[3]:.6f}"
                 for c, b in zip(cls_ids, bboxes)]
        self._save_raw(img, lines, stem, split)

    def _save_raw(self, img, lines, stem, split):
        (self.out / "images" / split / f"{stem}.jpg").parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(
            str(self.out / "images" / split / f"{stem}.jpg"),
            img, [cv2.IMWRITE_JPEG_QUALITY, cfg.JPEG_QUALITY]
        )
        lbl_path = self.out / "labels" / split / f"{stem}.txt"
        lbl_path.parent.mkdir(parents=True, exist_ok=True)
        lbl_path.write_text("\n".join(lines), encoding="utf-8")


# ═══════════════════════════════════════════════════════════════
# 10. TẠO data.yaml
# ═══════════════════════════════════════════════════════════════
def write_yaml(output_dir: Path, categories: list[str]):
    """
    Tạo data.yaml cho YOLO.
    Class = category (dùng cho classification/detection theo loại SP).
    """
    lines = [
        "# Dataset — YOLO Configuration",
        "",
        f"path: {output_dir.resolve()}",
        "train: images/train",
        "val:   images/val",
        "test:  images/test",
        "",
        f"nc: {len(categories)}",
        "names:",
    ]
    for i, cat in enumerate(sorted(categories)):
        lines.append(f"  {i}: {cat}")
    (output_dir / "data.yaml").write_text("\n".join(lines), encoding="utf-8")
    log.info(f"Đã tạo: {output_dir / 'data.yaml'}")


# ═══════════════════════════════════════════════════════════════
# 11. PIPELINE CHÍNH
# ═══════════════════════════════════════════════════════════════
class Preprocessor:

    def __init__(self):
        self.out = Path(cfg.OUTPUT_DIR)
        for split in ["train", "val", "test"]:
            (self.out / "images" / split).mkdir(parents=True, exist_ok=True)
            (self.out / "labels" / split).mkdir(parents=True, exist_ok=True)

    def run(self):
        log.info("═" * 55)
        log.info("  PREPROCESSING PIPELINE")
        log.info("═" * 55)

        # ── Bước 1: Parse tên file ──────────────────────────────
        log.info("Bước 1/5: Parse tên file...")
        df = load_dataset(cfg.INPUT_DIR)
        if df.empty:
            log.error("Không tìm thấy ảnh hợp lệ. Kiểm tra lại INPUT_DIR.")
            return
        print_stats(df)

        # ── Bước 2: Split theo product ─────────────────────────
        log.info("Bước 2/5: Stratified split theo product...")
        df = split_by_product(df)

        # ── Bước 3: Oversample ─────────────────────────────────
        if cfg.BALANCE:
            log.info("Bước 3/5: Cân bằng category (oversample train)...")
            df = oversample_train(df)
        else:
            log.info("Bước 3/5: Bỏ qua cân bằng.")

        # ── Bước 4: Xử lý ảnh ─────────────────────────────────
        log.info("Bước 4/5: Xử lý ảnh (clean → resize → augment)...")
        processor = SampleProcessor(self.out)
        rows      = df.to_dict("records")
        success   = 0

        with ThreadPoolExecutor(max_workers=cfg.NUM_WORKERS) as exe:
            futures = {exe.submit(processor.process, r): r for r in rows}
            for fut in tqdm(as_completed(futures), total=len(futures),
                            desc="  Processing"):
                if fut.result():
                    success += 1

        log.info(f"  Xử lý thành công: {success}/{len(rows)} ảnh")

        # ── Bước 5: Metadata + YAML ────────────────────────────
        log.info("Bước 5/5: Lưu metadata.csv + data.yaml...")
        df.drop(columns=["path"]).to_csv(
            self.out / "metadata.csv", index=False, encoding="utf-8"
        )
        write_yaml(self.out, df["category"].unique().tolist())

        self._verify()

    def _verify(self):
        print("\n" + "═" * 50)
        print("  KẾT QUẢ SAU TIỀN XỬ LÝ")
        print("═" * 50)
        total = 0
        for split in ["train", "val", "test"]:
            n = len(list((self.out / "images" / split).glob("*.jpg")))
            total += n
            print(f"  {split:<6}: {n:>5} ảnh")
        print(f"  {'TOTAL':<6}: {total:>5} ảnh")
        print(f"  metadata.csv : {'✅' if (self.out / 'metadata.csv').exists() else '❌'}")
        print(f"  data.yaml    : {'✅' if (self.out / 'data.yaml').exists() else '❌'}")
        print("═" * 50 + "\n")


# ═══════════════════════════════════════════════════════════════
# 12. VISUALIZE — Kiểm tra mẫu sau xử lý
# ═══════════════════════════════════════════════════════════════
def visualize_product(product_key: str, split: str = "train"):
    """
    Hiển thị tất cả các mặt của 1 sản phẩm sau khi xử lý.
    product_key ví dụ: 'meat.2'
    """
    out_dir = Path(cfg.OUTPUT_DIR) / "images" / split
    pattern = f"{product_key.replace('.', '.')}.*\\.jpg"
    imgs    = sorted(out_dir.glob(f"{product_key}.*.jpg"))

    if not imgs:
        print(f"Không tìm thấy ảnh cho '{product_key}' trong tập {split}")
        return

    fig, axes = plt.subplots(1, len(imgs), figsize=(5 * len(imgs), 5))
    if len(imgs) == 1:
        axes = [axes]

    for ax, img_path in zip(axes, imgs):
        img = cv2.imread(str(img_path))
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax.set_title(img_path.name, fontsize=9)
        ax.axis("off")

    plt.suptitle(f"Product: {product_key} [{split}]",
                 fontweight="bold", fontsize=12)
    plt.tight_layout()
    plt.savefig(f"check_{product_key}.png", dpi=150)
    plt.show()


# ═══════════════════════════════════════════════════════════════
# 13. ENTRY POINT
# ═══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Preprocessing — category.product_idx.face_idx.jpg"
    )
    parser.add_argument("--input-dir",    default=cfg.INPUT_DIR)
    parser.add_argument("--label-dir",    default=cfg.LABEL_DIR)
    parser.add_argument("--output-dir",   default=cfg.OUTPUT_DIR)
    parser.add_argument("--size",         type=int, default=cfg.TARGET_SIZE)
    parser.add_argument("--no-augment",   action="store_true")
    parser.add_argument("--no-balance",   action="store_true")
    parser.add_argument("--analyze-only", action="store_true",
                        help="Chỉ parse + thống kê, không xử lý ảnh")
    parser.add_argument("--plot",         action="store_true",
                        help="Vẽ biểu đồ phân phối")
    parser.add_argument("--check",        metavar="PRODUCT_KEY",
                        help="Visualize các mặt của 1 SP, ví dụ: meat.2")
    args = parser.parse_args()

    cfg.INPUT_DIR    = args.input_dir
    cfg.LABEL_DIR    = args.label_dir
    cfg.OUTPUT_DIR   = args.output_dir
    cfg.TARGET_SIZE  = args.size
    cfg.AUGMENT_TRAIN = not args.no_augment
    cfg.BALANCE       = not args.no_balance

    if args.check:
        visualize_product(args.check)

    elif args.analyze_only:
        df = load_dataset(cfg.INPUT_DIR)
        print_stats(df)
        if args.plot:
            plot_stats(df)

    else:
        Preprocessor().run()
        if args.plot:
            df = pd.read_csv(Path(cfg.OUTPUT_DIR) / "metadata.csv")
            plot_stats(df)
