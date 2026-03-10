"""
图像处理基础脚本
功能：读取图片 → 去畸变 → 鼠标选4点 → 逆透视变换 → 展示结果 → 导出映射表
"""
import os
import sys
import cv2
import numpy as np

# ============================================================
# 1. 硬编码相机内参矩阵 K 和畸变系数 D（伪造值，用于测试）
# ============================================================
# K: 3x3 内参矩阵，焦距约为 800 像素，光心假设在 (320, 240)
K = np.array([
    [800.0,   0.0, 320.0],
    [  0.0, 800.0, 240.0],
    [  0.0,   0.0,   1.0]
], dtype=np.float64)

# D: 5 个畸变系数 [k1, k2, p1, p2, k3]，设为微小值模拟轻微桶形畸变
D = np.array([-0.05, 0.01, 0.0, 0.0, 0.0], dtype=np.float64)


# ============================================================
# 2. 去畸变函数（参考 camera.py 的 _undistort 方法）
# ============================================================
def undistort(frame: np.ndarray, camera_matrix: np.ndarray, dist_coeffs: np.ndarray) -> np.ndarray:
    """对输入图像进行去畸变处理"""
    h, w = frame.shape[:2]
    # alpha=1 保留所有原始像素
    new_matrix, roi = cv2.getOptimalNewCameraMatrix(
        camera_matrix, dist_coeffs, (w, h), 1, (w, h)
    )
    undistorted = cv2.undistort(frame, camera_matrix, dist_coeffs, None, new_matrix)
    return undistorted


# ============================================================
# 3. 映射表计算与导出（参考 MATLAB 脚本的数学逻辑）
# ============================================================
def compute_undist_inverse_map(
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    M: np.ndarray,
    img_size: tuple,
) -> tuple:
    """
    计算去畸变 + 逆透视的组合映射表。
    对原图每个像素 (H, W)，先查去畸变映射，再代入透视矩阵 M。
    返回 (map_h, map_w)，形状均为 [H][W] 的二维数组。
    """
    h, w = img_size

    # 生成像素网格 (W, H)，与 MATLAB 中 meshgrid 对应
    ws, hs = np.meshgrid(np.arange(w, dtype=np.float64),
                         np.arange(h, dtype=np.float64))
    points = np.stack([ws.ravel(), hs.ravel()], axis=-1)  # shape: (H*W, 2)

    # 使用 cv2.undistortPoints 计算去畸变后的归一化坐标，再投影回像素坐标
    new_matrix, _ = cv2.getOptimalNewCameraMatrix(
        camera_matrix, dist_coeffs, (w, h), 1, (w, h)
    )
    undist_pts = cv2.undistortPoints(
        points.reshape(-1, 1, 2), camera_matrix, dist_coeffs, P=new_matrix
    )
    undist_pts = undist_pts.reshape(-1, 2)  # (H*W, 2), 每行 (x, y)

    undist_w = undist_pts[:, 0]  # 对应 MATLAB 的 UndistMapW
    undist_h = undist_pts[:, 1]  # 对应 MATLAB 的 UndistMapH

    # 将去畸变坐标代入透视矩阵 M（齐次变换）
    # MapH_i = (M[1,0]*W + M[1,1]*H + M[1,2]) / (M[2,0]*W + M[2,1]*H + 1)
    # MapW_i = (M[0,0]*W + M[0,1]*H + M[0,2]) / (M[2,0]*W + M[2,1]*H + 1)
    denom = M[2, 0] * undist_w + M[2, 1] * undist_h + 1.0
    map_h = (M[1, 0] * undist_w + M[1, 1] * undist_h + M[1, 2]) / denom
    map_w = (M[0, 0] * undist_w + M[0, 1] * undist_h + M[0, 2]) / denom

    return map_h.reshape(h, w), map_w.reshape(h, w)


def export_table_python(
    map_h: np.ndarray,
    map_w: np.ndarray,
    name_h: str,
    name_w: str,
    out_dir: str,
) -> None:
    """
    将映射表导出为 .py 文件（Python 二维列表格式）。
    格式对齐 MATLAB 的 PrintTablePython 函数。
    """
    os.makedirs(out_dir, exist_ok=True)
    h, w = map_h.shape

    for data, name in [(map_h, name_h), (map_w, name_w)]:
        filepath = os.path.join(out_dir, f"{name}.py")
        with open(filepath, "w") as f:
            f.write(f"{name} = [\n")
            for row_idx in range(h):
                vals = ", ".join(f"{data[row_idx, col]:.2f}" for col in range(w))
                comma = "," if row_idx < h - 1 else ""
                f.write(f"    [{vals}]{comma}\n")
            f.write("]\n")
        print(f"Exported: {filepath}")


def export_table_c(
    map_h: np.ndarray,
    map_w: np.ndarray,
    name_h: str,
    name_w: str,
    out_dir: str,
) -> None:
    """
    将映射表导出为 .txt 文件（C 语言 const float 数组格式）。
    格式对齐 MATLAB 的 PrintTable 函数。
    """
    os.makedirs(out_dir, exist_ok=True)
    h, w = map_h.shape

    for data, name in [(map_h, name_h), (map_w, name_w)]:
        filepath = os.path.join(out_dir, f"{name}.txt")
        with open(filepath, "w") as f:
            f.write(f"const float {name}[{h}][{w}] = {{\n")
            for row_idx in range(h):
                vals = ", ".join(f"{data[row_idx, col]:.2f}" for col in range(w))
                comma = "," if row_idx < h - 1 else ""
                f.write(f"{{{vals}}}{comma}\n")
            f.write("};\n")
        print(f"Exported: {filepath}")


# ============================================================
# 4. 全局状态：用于鼠标回调
# ============================================================
clicked_points = []           # 存储用户点击的坐标
undistorted_img = None        # 去畸变后的图像（用于绘制和变换）
undistorted_img_display = None  # 去畸变图像的显示副本（可在上面画点）
original_img = None           # 原图引用（供打表时获取尺寸）

WIN_UNDISTORTED = "Undistorted (click 4 points)"
WIN_PERSPECTIVE = "Inverse Perspective"
OUTPUT_DIR = "PrintTable"     # 导出目录


# ============================================================
# 5. 鼠标回调（参考 NEUQVisionTransformAPP2.py）
# ============================================================
def on_mouse_click(event, x, y, flags, param):
    """鼠标左键点击回调：收集 4 个点后执行逆透视变换并导出映射表"""
    global clicked_points, undistorted_img, undistorted_img_display, original_img

    if event != cv2.EVENT_LBUTTONDOWN:
        return
    if len(clicked_points) >= 4:
        return

    # 记录坐标
    clicked_points.append([x, y])

    # 在图像上绘制标记
    cv2.circle(undistorted_img_display, (x, y), 5, (0, 0, 255), -1)
    cv2.putText(
        undistorted_img_display,
        f"({x}, {y})",
        (x + 8, y - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 255),
        1,
    )
    cv2.imshow(WIN_UNDISTORTED, undistorted_img_display)
    print(f"Point {len(clicked_points)}: ({x}, {y})")

    # 收集满 4 个点后，计算透视变换并显示结果
    if len(clicked_points) == 4:
        print("4 points collected, computing perspective transform...")

        pts_src = np.float32(clicked_points)

        # 目标矩形：根据源点的包围框大小确定输出尺寸
        x_min = min(p[0] for p in clicked_points)
        x_max = max(p[0] for p in clicked_points)
        y_min = min(p[1] for p in clicked_points)
        y_max = max(p[1] for p in clicked_points)
        out_w = x_max - x_min
        out_h = y_max - y_min

        # 目标四角（标准矩形）
        margin = 10
        pts_dst = np.float32([
            [margin,         margin],
            [out_w + margin, margin],
            [margin,         out_h + margin],
            [out_w + margin, out_h + margin],
        ])

        M = cv2.getPerspectiveTransform(pts_src, pts_dst)
        print(f"Perspective Matrix M:\n{M}")

        result = cv2.warpPerspective(
            undistorted_img, M, (out_w + 2 * margin, out_h + 2 * margin)
        )
        cv2.imshow(WIN_PERSPECTIVE, result)

        # --- 计算并导出映射表 ---
        img_size = (original_img.shape[0], original_img.shape[1])
        print(f"Computing undist+inverse mapping table for {img_size[1]}x{img_size[0]}...")
        map_h, map_w = compute_undist_inverse_map(K, D, M, img_size)

        # 导出 Python 格式
        export_table_python(map_h, map_w,
                            "UndistInverseMapH", "UndistInverseMapW", OUTPUT_DIR)
        # 导出 C 格式
        export_table_c(map_h, map_w,
                       "UndistInverseMapH", "UndistInverseMapW", OUTPUT_DIR)
        print("All mapping tables exported.")


# ============================================================
# 6. 主流程
# ============================================================
def main():
    global undistorted_img, undistorted_img_display, original_img

    # 默认图片路径，支持命令行参数指定
    img_path = sys.argv[1] if len(sys.argv) > 1 else "VisionTransform/TestImage/img.jpg"

    # 读取图片
    img = cv2.imread(img_path)
    if img is None:
        print(f"Error: cannot read image '{img_path}'")
        sys.exit(1)
    print(f"Loaded image: {img_path}  ({img.shape[1]}x{img.shape[0]})")

    # 保存原图引用（供打表使用）
    original_img = img

    # 显示原图
    cv2.imshow("Original", img)

    # 去畸变
    undistorted_img = undistort(img, K, D)
    undistorted_img_display = undistorted_img.copy()

    # 显示去畸变图并绑定鼠标回调
    cv2.namedWindow(WIN_UNDISTORTED)
    cv2.setMouseCallback(WIN_UNDISTORTED, on_mouse_click)
    cv2.imshow(WIN_UNDISTORTED, undistorted_img_display)

    print("Please click 4 points on the undistorted image window.")
    print("Press any key to exit after viewing results.")

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
