import os
import sys
import pandas as pd
import matplotlib.pyplot as plt

def count_and_plot(base_path):
    """
    Đếm số lượng file trong các thư mục con và vẽ biểu đồ.
    """
    
    # 1. Kiểm tra xem các thư viện cần thiết đã được cài đặt chưa
    try:
        import pandas as pd
        import matplotlib.pyplot as plt
    except ImportError:
        print("+" * 50)
        print("LỖI: Bạn cần cài đặt thư viện 'pandas' và 'matplotlib'.")
        print("Hãy chạy lệnh sau trong PowerShell hoặc CMD:")
        print("pip install pandas matplotlib")
        print("+" * 50)
        sys.exit()

    # 2. Đếm số lượng file
    class_counts = []
    
    try:
        print(f"Bắt đầu quét thư mục: {base_path}")
        # Lấy danh sách tất cả các mục (thư mục/file) trong đường dẫn gốc
        all_items_in_base = os.listdir(base_path)
    except FileNotFoundError:
        print(f"LỖI: Không tìm thấy đường dẫn '{base_path}'.")
        print("Vui lòng kiểm tra lại đường dẫn.")
        return
    except Exception as e:
        print(f"LỖI khi truy cập {base_path}: {e}")
        return

    if not all_items_in_base:
        print(f"Thư mục '{base_path}' bị rỗng.")
        return

    print(f"Tìm thấy {len(all_items_in_base)} mục. Bắt đầu lọc và đếm file...")
    
    # Lặp qua từng mục
    for item_name in all_items_in_base:
        item_path = os.path.join(base_path, item_name)
        
        # Chỉ xử lý nếu mục đó là một THƯ MỤC
        if os.path.isdir(item_path):
            try:
                # Đếm tất cả các file trong thư mục con này
                files = [f for f in os.listdir(item_path) if os.path.isfile(os.path.join(item_path, f))]
                num_files = len(files)
                class_counts.append({'class_name': item_name, 'count': num_files})
            except Exception as e:
                print(f"-> Không thể đếm file trong thư mục '{item_path}': {e}")
    
    if not class_counts:
        print("Không tìm thấy thư mục con nào (lớp) bên trong đường dẫn chính.")
        return
        
    # 3. Tạo DataFrame (bảng dữ liệu) từ kết quả
    df_counts = pd.DataFrame(class_counts)
    
    # Sắp xếp dữ liệu (số lượng nhiều nhất lên trên)
    df_counts = df_counts.sort_values(by='count', ascending=False)
    
    # In ra bảng đếm
    print("\n--- Thống kê số lượng ảnh mỗi lớp ---")
    print(df_counts.to_string()) # .to_string() để đảm bảo in ra toàn bộ
    print("-" * 40)
    
    # 4. Vẽ biểu đồ
    
    num_classes = len(df_counts)
    
    # Điều chỉnh kích thước biểu đồ (chiều cao) dựa trên số lượng lớp
    fig_height = max(6, num_classes * 0.45) # Tối thiểu 6 inch
    fig_width = 10 # Chiều rộng 10 inch
    
    plt.figure(figsize=(fig_width, fig_height))
    
    # Vẽ biểu đồ thanh ngang (barh) vì nó dễ đọc tên các lớp hơn
    plt.barh(df_counts['class_name'], df_counts['count'], color='skyblue')
    
    # Đảo ngược trục y để lớp có số lượng cao nhất ở trên cùng
    plt.gca().invert_yaxis()
    
    plt.xlabel('Số lượng ảnh')
    plt.ylabel('Lớp (Class)')
    plt.title('Số lượng ảnh trong mỗi lớp')
    
    # Thêm đường lưới
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    
    # Tự động điều chỉnh lề để nhãn không bị cắt
    plt.tight_layout()
    
    # 5. Lưu biểu đồ
    plot_filename = 'thong_ke_so_luong_anh.png'
    try:
        plt.savefig(plot_filename)
        print(f"\nĐã lưu biểu đồ vào file: {plot_filename}")
        print(f"Bạn có thể mở file '{plot_filename}' để xem kết quả trực quan.")
    except Exception as e:
        print(f"LỖI khi lưu biểu đồ: {e}")

# --- PHẦN THỰC THI CHÍNH ---

# 1. Định nghĩa đường dẫn
# !!! QUAN TRỌNG: Đảm bảo đường dẫn này chính xác !!!
target_directory = r"C:/Users/tam/Desktop/Data/leaf/plantvillage dataset"

# 2. Chạy hàm
count_and_plot(target_directory)