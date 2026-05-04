"""
图像处理基础脚本
功能：读取图片 → 去畸变 → 鼠标选4点 → 逆透视变换 → 展示结果 → 导出映射表
"""
import os
import sys
import cv2
import numpy as np

from history_manager import HistoryManager

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

    K_new = K.copy().astype(np.float64)
    K_new[0, 0] *= scale_x  # 缩放 fx
    K_new[0, 2] *= scale_x  # 缩放 cx
    K_new[1, 1] *= scale_y  # 缩放 fy
    K_new[1, 2] *= scale_y  # 缩放 cy
    
    return K_new


def order_points(pts: np.ndarray) -> np.ndarray:
    """Return points ordered as top-left, top-right, bottom-left, bottom-right."""
    pts = np.asarray(pts, dtype=np.float32).reshape(-1, 2)
    if pts.shape[0] != 4:
        raise ValueError("Exactly 4 points are required")
    if len(np.unique(pts, axis=0)) != 4:
        raise ValueError("Perspective points must be unique")

    y_sorted = pts[np.argsort(pts[:, 1])]
    top = y_sorted[:2]
    bottom = y_sorted[2:]

    top = top[np.argsort(top[:, 0])]
    bottom = bottom[np.argsort(bottom[:, 0])]

    top_left, top_right = top
    bottom_left, bottom_right = bottom

    return np.float32([top_left, top_right, bottom_left, bottom_right])


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
def get_undistort_matrices(
    K: np.ndarray,
    D: np.ndarray,
    image_size: tuple,
    calib_size: tuple = (640, 480),
    model: str = "pinhole",
) -> tuple:
    """生成去畸变预览和打表必须共用的 K_work / new_K。"""
    w, h = image_size
    if (w, h) != calib_size:
        K_work = scale_camera_matrix(K, calib_size, (w, h))
    else:
        K_work = K.copy().astype(np.float64)

    D_work = D.reshape(-1, 1).astype(np.float64)
    if model == "fisheye":
        new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
            K_work, D_work, (w, h), np.eye(3), balance=1.0, new_size=(w, h)
        )
    else:
        new_K, _ = cv2.getOptimalNewCameraMatrix(K_work, D_work, (w, h), 1, (w, h))

    return K_work, new_K


def undistort(
    img: np.ndarray,
    K: np.ndarray,
    D: np.ndarray,
    calib_size: tuple = (640, 480),
    model: str = "pinhole",
    return_matrices: bool = False,
) -> np.ndarray:
    """
    去畸变函数 (自带分辨率自适应缩放功能)
    calib_size: 默认该 K 矩阵是在 640x480 分辨率下标定得出的。
    """
    h, w = img.shape[:2]
    D_work = D.reshape(-1, 1).astype(np.float64)
    K_work, new_K = get_undistort_matrices(K, D_work, (w, h), calib_size, model)

    if model == "fisheye":
        map_x, map_y = cv2.fisheye.initUndistortRectifyMap(
            K_work, D_work, np.eye(3), new_K, (w, h), cv2.CV_32FC1
        )
        undist_img = cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR)
    else:
        undist_img = cv2.undistort(img, K_work, D_work, None, new_K)

    if return_matrices:
        return undist_img, K_work, new_K
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
    new_camera_matrix: np.ndarray = None,
    model: str = "pinhole",
) -> tuple:
    """生成反向映射表，带默认尺寸防御，防止传参错位"""
    out_w, out_h = out_size
    img_w, img_h = img_size
    D_work = dist_coeffs.reshape(-1, 1).astype(np.float64)
    K_work = camera_matrix.astype(np.float64)
    if new_camera_matrix is None:
        new_camera_matrix = K_work
    else:
        new_camera_matrix = new_camera_matrix.astype(np.float64)

    if model == "fisheye":
        map_x, map_y = cv2.fisheye.initUndistortRectifyMap(
            K_work, D_work, np.eye(3), new_camera_matrix, (img_w, img_h), cv2.CV_32FC1
        )
    else:
        map_x, map_y = cv2.initUndistortRectifyMap(
            K_work, D_work, None, new_camera_matrix, (img_w, img_h), cv2.CV_32FC1
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
    bev_size: tuple = None,
    new_camera_matrix: np.ndarray = None,
    model: str = "pinhole",
) -> tuple:
    """
    生成前向点表: 输入图点 (x, y) -> BEV 点 (u, v) (自带分辨率自适应)
    """
    if bev_size is None:
        bev_size = in_size
    in_w, in_h = in_size
    D_work = dist_coeffs.reshape(-1, 1).astype(np.float64)

    # 1. 核心修复：动态缩放 K 矩阵到下位机巡线尺寸 in_size
    K_in = scale_camera_matrix(camera_matrix, calib_size, in_size)
    if new_camera_matrix is None:
        _, new_camera_matrix = get_undistort_matrices(
            camera_matrix, D_work, img_size, calib_size, model
        )
    else:
        new_camera_matrix = new_camera_matrix.astype(np.float64)

    # 2. 生成 in_size (160x120) 的网格坐标
    xs, ys = np.meshgrid(
        np.arange(in_w, dtype=np.float32),
        np.arange(in_h, dtype=np.float32)
    )
    pts_in = np.stack([xs, ys], axis=-1).reshape(-1, 1, 2)

    # 3. 直接把输入图点去畸变到 GUI 预览使用的 new_K 坐标系
    if model == "fisheye":
        undist_pts_img = cv2.fisheye.undistortPoints(
            pts_in, K_in, D_work, P=new_camera_matrix
        )
    else:
        undist_pts_img = cv2.undistortPoints(
            pts_in, K_in, D_work, P=new_camera_matrix
        )

    # 5. 施加透视变换矩阵 M
    bev_pts = cv2.perspectiveTransform(undist_pts_img, M).reshape(-1, 2)

    map_w = bev_pts[:, 0].reshape(in_h, in_w).astype(np.float32)
    map_h = bev_pts[:, 1].reshape(in_h, in_w).astype(np.float32)

    # 6. 掩码与越界钳位 (-1 哨兵值保护)
    bev_w, bev_h = bev_size
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
working_K = None              # 当前图片尺寸下的内参矩阵
preview_K = None              # 去畸变预览使用的新内参矩阵
calib_size = (640, 480)
calib_model = "pinhole"

WIN_UNDISTORTED = "Undistorted (click 4 points)"
WIN_PERSPECTIVE = "Inverse Perspective"
OUTPUT_DIR = "PrintTable"     # 导出目录


# ============================================================
# 5. 鼠标回调（参考 NEUQVisionTransformAPP2.py）
# ============================================================
def on_mouse_click(event, x, y, flags, param):
    """鼠标左键点击回调：收集 4 个点后执行逆透视变换并导出映射表"""
    global clicked_points, undistorted_img, undistorted_img_display, original_img
    global working_K, preview_K, calib_model

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

        pts_src = order_points(clicked_points)

        # 目标矩形：根据源点的包围框大小确定输出尺寸
        x_min = min(p[0] for p in clicked_points)
        x_max = max(p[0] for p in clicked_points)
        y_min = min(p[1] for p in clicked_points)
        y_max = max(p[1] for p in clicked_points)
        out_w = x_max - x_min
        out_h = y_max - y_min

        out_w = max(1, int(round(out_w)))
        out_h = max(1, int(round(out_h)))

        # 目标四角（标准矩形，无额外 margin，保证预览和导出表坐标一致）
        pts_dst = np.float32([
            [0,         0],
            [out_w - 1, 0],
            [0,         out_h - 1],
            [out_w - 1, out_h - 1],
        ])

        M = cv2.getPerspectiveTransform(pts_src, pts_dst)
        print(f"Perspective Matrix M:\n{M}")

        result = cv2.warpPerspective(
            undistorted_img, M, (out_w, out_h)
        )
        cv2.imshow(WIN_PERSPECTIVE, result)

        # --- 计算并导出映射表 ---
        img_size = (original_img.shape[1], original_img.shape[0])
        print(f"Computing backward mapping table for {img_size[0]}x{img_size[1]}...")
        map_h, map_w = compute_backward_remap(
            camera_matrix=working_K if working_K is not None else K,
            dist_coeffs=D,
            M=M,
            out_size=(out_w, out_h),
            img_size=img_size,
            new_camera_matrix=preview_K,
            model=calib_model,
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
    global K, D, undistorted_img, undistorted_img_display, original_img
    global working_K, preview_K, calib_size, calib_model

    # 默认图片路径，支持命令行参数指定
    img_path = sys.argv[1] if len(sys.argv) > 1 else "VisionTransform/TestImage/img.jpg"
    calib_path = sys.argv[2] if len(sys.argv) > 2 else None

    if calib_path:
        bundle = HistoryManager.import_calibration_bundle_from_yaml(calib_path)
        K = bundle["K"].astype(np.float64)
        D = bundle["D"].reshape(-1, 1).astype(np.float64)
        calib_model = bundle.get("model", "pinhole")
        calib_size = bundle.get("image_size") or calib_size
        print(f"Loaded calibration: {calib_path} ({calib_model}, {calib_size[0]}x{calib_size[1]})")

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
    undistorted_img, working_K, preview_K = undistort(
        img, K, D, calib_size=calib_size, model=calib_model, return_matrices=True
    )
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
