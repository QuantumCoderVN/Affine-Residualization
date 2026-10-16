"""Chỉ hai việc: so sánh nghiệm, rồi so sánh thời gian giải.

Chạy: python main.py
Kết quả in ra màn hình và lưu trong một ảnh: so_sanh.png.
Không phụ thuộc bất kỳ file nào của bản triển khai trước.
"""
from pathlib import Path
from time import perf_counter
from contextlib import nullcontext
import numpy as np
from scipy.linalg import solve, cho_factor, cho_solve
from scipy.sparse.linalg import cg
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# Có thể thay đổi các thông số này rồi chạy lại.
N_SO_SANH_NGHIEM = 128
CAC_KICH_THUOC = [32, 64, 128, 256, 512]
KAPPA = 1024.0
SO_LAN_DO = 11
TOL = 1e-10
SEED = 2026


def tao_he(n):
    """Tạo hệ Ax=b đối xứng xác định dương để dùng được cả CG và Cholesky.

    Hệ có một hướng khó q0, với trị riêng lambda0=1/KAPPA.
    Trong mô hình này q0,lambda0 được biết sẵn, như cấu trúc bổ sung trong PDF.
    Chúng được trả riêng; các solver không nhận nghiệm dùng để sinh dữ liệu.
    """
    rng = np.random.default_rng(SEED + n)
    Q, _ = np.linalg.qr(rng.normal(size=(n, n)))
    tri_rieng = np.r_[1.0 / KAPPA, np.geomspace(0.2, 1.0, n - 1)]
    A = (Q * tri_rieng) @ Q.T
    he_so = np.r_[1.0, rng.normal(size=n - 1)]
    he_so[1:] /= np.linalg.norm(he_so[1:])
    b = A @ (Q @ he_so)
    b /= np.linalg.norm(b)
    return A, b, Q[:, 0], tri_rieng[0]


def giai_cg(A, b, nguong):
    """CG với ngưỡng residual tuyệt đối được truyền vào rõ ràng."""
    x, trang_thai = cg(A, b, rtol=0.0, atol=nguong, maxiter=10 * len(b))
    if trang_thai != 0:
        raise RuntimeError(f"CG chưa hội tụ, info={trang_thai}")
    return x


def giai_affine(A, b, q0, lambda0, nguong):
    """Ý tưởng trong PDF: giải phần chưa biết rồi cộng lại.

    1. z là phần nghiệm trên hướng khó đã biết: z=q0*(q0.T@b)/lambda0.
    2. Tính phần dư r=b-Az.
    3. Giải Ay=r bằng CG, thu được y.
    4. Trả x=z+y. Không sử dụng nghiệm cổ điển để xây z hoặc y.
    """
    z = q0 * (q0 @ b) / lambda0
    r = b - A @ z
    y = giai_cg(A, r, nguong)
    return z + y


def tao_cac_bo_giai(A, b, q0, lambda0):
    # Cùng ngưỡng residual cuối cho CG trực tiếp và affine-CG.
    nguong = TOL * np.linalg.norm(b)
    return {
        "LU (cổ điển)": lambda: solve(A, b, assume_a="gen", check_finite=False),
        "Cholesky": lambda: cho_solve(cho_factor(A, check_finite=False), b,
                                     check_finite=False),
        "CG": lambda: giai_cg(A, b, nguong),
        "Affine + CG": lambda: giai_affine(A, b, q0, lambda0, nguong),
    }


def sai_so_tuong_doi(x, x_tham_chieu):
    return np.linalg.norm(x - x_tham_chieu) / np.linalg.norm(x_tham_chieu)


def chay():
    # Giới hạn BLAS về một thread để bớt nhiễu do cách chia luồng.
    try:
        from threadpoolctl import threadpool_limits
        gioi_han_luong = threadpool_limits(limits=1)
    except ImportError:
        gioi_han_luong = nullcontext()

    with gioi_han_luong:
        print("\n1. HAI CÁCH GIẢI CÓ CHO CÙNG NGHIỆM KHÔNG?")
        A, b, q0, lambda0 = tao_he(N_SO_SANH_NGHIEM)
        cac_bo_giai = tao_cac_bo_giai(A, b, q0, lambda0)
        x_co_dien = cac_bo_giai["LU (cổ điển)"]()
        x_affine = cac_bo_giai["Affine + CG"]()
        loi = sai_so_tuong_doi(x_affine, x_co_dien)
        print(f"Hệ Ax=b: n={N_SO_SANH_NGHIEM}, kappa={KAPPA:g}.")
        print("Cổ điển: giải trực tiếp Ax=b bằng LU.")
        print("Affine: tính z, r=b-Az; giải Ay=r bằng CG; trả x=z+y.")
        print("Metric: E = ||x_affine - x_co_dien||₂ / ||x_co_dien||₂.")
        print(f"Sai số tương đối E = {loi:.6e}.")
        print("E gần 0 nghĩa là hai nghiệm gần trùng nhau.")
        print("\n10 thành phần đầu (biểu đồ vẽ tất cả các thành phần):")
        print(f"{'i':>4} {'Nghiệm cổ điển':>20} {'Nghiệm affine':>20}")
        for i in range(min(10, len(b))):
            print(f"{i+1:4d} {x_co_dien[i]:20.10f} {x_affine[i]:20.10f}")

        print("\n2. CÁCH NÀO GIẢI NHANH HƠN TRÊN CPU?")
        print(f"Metric: thời gian median (ms), qua {SO_LAN_DO} lần đo; thấp hơn là nhanh hơn.")
        print("Tạo ma trận không tính vào timer của phương pháp nào.")
        print("Mỗi lần giải LU/Cholesky đều phân tích lại ma trận.")
        print("Affine tính cả thời gian tạo z, tính r, giải y và cộng z+y.")
        print("Việc tìm q0 không được đo: đây là cấu trúc biết sẵn của mô hình.")
        thoi_gian = []
        ten_bo_giai = list(cac_bo_giai)
        print(f"\n{'n':>5}" + "".join(f"{ten:>20}" for ten in ten_bo_giai))
        for n in CAC_KICH_THUOC:
            A_n, b_n, q_n, lambda_n = tao_he(n)
            bo_giai = tao_cac_bo_giai(A_n, b_n, q_n, lambda_n)
            x_chuan = bo_giai["LU (cổ điển)"]()
            mau = {ten: [] for ten in ten_bo_giai}
            for ham in bo_giai.values():
                ham()  # warmup; không đưa vào thời gian đo
            rng = np.random.default_rng(SEED)
            for _ in range(SO_LAN_DO):
                # Đổi thứ tự đo để giảm thiên lệch do phương pháp chạy trước/sau.
                for ten in rng.permutation(ten_bo_giai):
                    bat_dau = perf_counter()
                    x = bo_giai[ten]()
                    dt_ms = (perf_counter() - bat_dau) * 1000
                    mau[ten].append(dt_ms)
                    # Kiểm tra ngoài timer: không chấp nhận nghiệm sai để lấy tốc độ.
                    if sai_so_tuong_doi(x, x_chuan) > 1e-7:
                        raise RuntimeError(f"Nghiệm {ten} tại n={n} chưa đủ chính xác")
            hang = {ten: float(np.median(mau[ten])) for ten in ten_bo_giai}
            thoi_gian.append(hang)
            print(f"{n:5d}" + "".join(f"{hang[ten]:20.5f}" for ten in ten_bo_giai))

    # Chỉ một ảnh, có đúng hai phần tương ứng hai câu hỏi.
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11})
    fig, axes = plt.subplots(2, 1, figsize=(11.5, 9.2), layout="constrained")
    chi_so = np.arange(1, N_SO_SANH_NGHIEM + 1)
    axes[0].plot(chi_so, x_co_dien, color="#2869a5", lw=2.5, label="Nghiệm cổ điển (LU)")
    axes[0].plot(chi_so, x_affine, color="#e17b27", lw=1.8, ls="--", label="Nghiệm affine (z + y)")
    axes[0].set(title=f"1. So sánh nghiệm trên cùng hệ Ax=b, n={N_SO_SANH_NGHIEM}",
                xlabel="Chỉ số thành phần i", ylabel="Giá trị x[i]")
    axes[0].text(.02, .96, f"Sai số tương đối E = {loi:.2e}\nGần 0 → hai nghiệm gần trùng nhau",
                 transform=axes[0].transAxes, va="top", fontsize=11,
                 bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="#bac9d5", alpha=.95))
    axes[0].legend(loc="upper right")
    for ten in ten_bo_giai:
        axes[1].plot(CAC_KICH_THUOC, [hang[ten] for hang in thoi_gian], "o-", lw=2,
                     label=ten)
    axes[1].set(title=f"2. Thời gian giải trên CPU: median của {SO_LAN_DO} lần đo",
                xlabel=f"Kích thước ma trận n (cùng kappa={KAPPA:g})",
                ylabel="Thời gian (ms, thang log) — thấp hơn tốt hơn", yscale="log")
    axes[1].set_xticks(CAC_KICH_THUOC)
    axes[1].legend()
    for ax in axes:
        ax.grid(alpha=.2)
        ax.spines[["top", "right"]].set_visible(False)
    duong_dan = Path(__file__).resolve().with_name("so_sanh.png")
    fig.savefig(duong_dan, dpi=160)
    plt.close(fig)
    print(f"\nĐã lưu hai biểu đồ trong cùng một ảnh: {duong_dan}")
    print("Đây là mô hình kiểm chứng ý tưởng affine trên CPU; thời gian này chưa đại diện cho tốc độ thuật toán lượng tử.")


if __name__ == "__main__":
    chay()
