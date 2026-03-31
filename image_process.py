"""
图像处理基础脚本
功能：读取图片 → 去畸变 → 鼠标选4点 → 逆透视变换 → 展示结果 → 导出映射表
"""
import os
import sys
import cv2
import numpy as np

def scale_camera_matrix(K: np.ndarray, orig_size: tuple, new_size: tuple) -> np.ndarray:
    """
    按分辨率比例动态缩放相机内参 K。
    orig_size: (width, height) 标定时的分辨率，如 (640, 480)
    new_size:  (width, height) 当前图片的分辨率，如 (160, 120)
    """
    orig_w, orig_h = orig_size
    new_w, new_h = new_size

    scale_x = new_w / orig_w
    scale_y = new_h / orig_h

    K_new = K.copy().astype(np.float32)
    K_new[0, 0] *= scale_x  # 缩放 fx
    K_new[0, 2] *= scale_x  # 缩放 cx
    K_new[1, 1] *= scale_y  # 缩放 fy
    K_new[1, 2] *= scale_y  # 缩放 cy
    
    return K_new


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
def undistort(img: np.ndarray, K: np.ndarray, D: np.ndarray, calib_size: tuple = (640, 480)) -> np.ndarray:
    """
    去畸变函数 (自带分辨率自适应缩放功能)
    calib_size: 默认该 K 矩阵是在 640x480 分辨率下标定得出的。
    """
    h, w = img.shape[:2]
    
    # 1. 动态自适应 K 矩阵
    if (w, h) != calib_size:
        K_work = scale_camera_matrix(K, calib_size, (w, h))
    else:
        K_work = K.copy()

    # 2. 计算新的最优内参 (保留全图视野，alpha=1)
    new_K, roi = cv2.getOptimalNewCameraMatrix(K_work, D, (w, h), 1, (w, h))
    
    # 3. 执行真正的去畸变 (注意这里传入的是 K_work 而不是旧的 K)
    undist_img = cv2.undistort(img, K_work, D, None, new_K)
    
    return undist_img


# ============================================================
# 3. 映射表计算与导出（参考 MATLAB 脚本的数学逻辑）
# ============================================================
def compute_backward_remap(
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    M: np.ndarray,
    out_size: tuple = (160, 120),
    img_size: tuple = (640, 480),
) -> tuple:
    """生成反向映射表，带默认尺寸防御，防止传参错位"""
    out_w, out_h = out_size
    img_w, img_h = img_size

    map_x, map_y = cv2.initUndistortRectifyMap(
        camera_matrix, dist_coeffs, None, camera_matrix, (img_w, img_h), cv2.CV_32FC1
    )

    u, v = np.meshgrid(np.arange(out_w, dtype=np.float32),
                       np.arange(out_h, dtype=np.float32))
    bev_pts = np.stack([u, v], axis=-1).reshape(-1, 1, 2)

    M_inv = np.linalg.inv(M)
    undist_pts = cv2.perspectiveTransform(bev_pts, M_inv)

    undist_x = undist_pts[:, 0, 0].reshape(out_h, out_w).astype(np.float32)
    undist_y = undist_pts[:, 0, 1].reshape(out_h, out_w).astype(np.float32)

    # 越界奇点填为 -1，C++ 手写循环遇到负数直接填黑即可
    dist_x = cv2.remap(map_x, undist_x, undist_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=-1)
    dist_y = cv2.remap(map_y, undist_x, undist_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=-1)

    return dist_y, dist_x


def compute_forward_point_map(
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    M: np.ndarray,
    in_size: tuple = (160, 120),  # 下位机 C++ 巡线时的真实输入分辨率
    img_size: tuple = (640, 480), # 你在 GUI 里加载测试图片的真实分辨率
    calib_size: tuple = (640, 480), # 最初相机的标定分辨率 (默认 640x480)
    bev_size: tuple = None
) -> tuple:
    """
    生成前向点表: 输入图点 (x, y) -> BEV 点 (u, v) (自带分辨率自适应)
    """
    if bev_size is None:
        bev_size = in_size
    in_w, in_h = in_size
    img_w, img_h = img_size

    # 1. 核心修复：动态缩放 K 矩阵到下位机巡线尺寸 in_size
    K_in = scale_camera_matrix(camera_matrix, calib_size, in_size)
    new_K_in, _ = cv2.getOptimalNewCameraMatrix(K_in, dist_coeffs, in_size, 1, in_size)

    # 2. 生成 in_size (160x120) 的网格坐标
    xs, ys = np.meshgrid(
        np.arange(in_w, dtype=np.float32),
        np.arange(in_h, dtype=np.float32)
    )
    pts_in = np.stack([xs, ys], axis=-1).reshape(-1, 1, 2)

    # 3. 使用【缩小后的 K 矩阵】在 160x120 空间下进行精准去畸变
    undist_pts_in = cv2.undistortPoints(pts_in, K_in, dist_coeffs, P=new_K_in)

    # 4. 将去畸变后的点，等比缩放到 M 矩阵所在的 GUI 截图空间
    # (如果 GUI 里加载的就是 160x120 图片，这里 scale 就是 1，完美兼容)
    scale_to_m_x = img_w / float(in_w)
    scale_to_m_y = img_h / float(in_h)
    undist_pts_img = undist_pts_in.copy()
    undist_pts_img[:, 0, 0] *= scale_to_m_x
    undist_pts_img[:, 0, 1] *= scale_to_m_y

    # 5. 施加透视变换矩阵 M
    bev_pts = cv2.perspectiveTransform(undist_pts_img, M).reshape(-1, 2)

    map_w = bev_pts[:, 0].reshape(in_h, in_w).astype(np.float32)
    map_h = bev_pts[:, 1].reshape(in_h, in_w).astype(np.float32)

    # 6. 掩码与越界钳位 (-1 哨兵值保护)
    bev_w, bev_h = in_size # C++ 中 BEV 画布尺寸与 in_size 一致
    valid = ((map_w >= 0) & (map_w < bev_w) & (map_h >= 0) & (map_h < bev_h))
    
    map_w[~valid] = -1.0
    map_h[~valid] = -1.0

    return map_h, map_w


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
                vals = ", ".join("{:.2f}".format(float(data[row_idx, col])) for col in range(w))
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
                vals = ", ".join("{:.2f}".format(float(data[row_idx, col])) for col in range(w))
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
        img_size = (original_img.shape[1], original_img.shape[0])
        print(f"Computing backward mapping table for {img_size[0]}x{img_size[1]}...")
        map_h, map_w = compute_backward_remap(
            camera_matrix=K,
            dist_coeffs=D,
            M=M,
            out_size=(out_w, out_h),
            img_size=img_size
        )

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
