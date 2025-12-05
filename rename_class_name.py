# import os
# print(os.listdir(r"C:/Users/tam/Desktop/Data/leaf/plantvillage dataset"))
# print(len(os.listdir(r"C:/Users/tam/Desktop/Data/leaf/plantvillage dataset")))
import os
import re

def standardize_name(name):
    """
    Chuẩn hóa tên (thư mục hoặc file):
    1. Thay thế dấu gạch dưới kép (___), dấu cách, dấu phẩy, và dấu ngoặc đơn () bằng dấu gạch dưới (_).
    2. Loại bỏ các ký tự đặc biệt khác.
    3. Chuyển tất cả về chữ thường.
    4. Xóa các dấu gạch dưới thừa.
    """
    # 1. Thay thế các dấu phân cách chính
    name = name.replace('___', '_').replace(' ', '_').replace('(', '').replace(')', '').replace(',', '')
    
    # 2. Loại bỏ các ký tự đặc biệt khác và chuyển về chữ thường
    # Giữ lại chữ cái, số, và dấu gạch dưới
    name = re.sub(r'[^\w_]', '', name).lower()
    
    # 4. Xóa các dấu gạch dưới thừa
    # Thay thế nhiều dấu gạch dưới liên tiếp bằng một dấu gạch dưới duy nhất
    name = re.sub(r'_{2,}', '_', name) 
    
    # Xóa dấu gạch dưới ở đầu và cuối
    name = name.strip('_')
    
    return name

def rename_directories(base_path):
    """
    Đổi tên tất cả các thư mục cấp 1 trong đường dẫn base_path 
    theo định dạng chuẩn hóa (lowercase, gạch dưới).
    """
    try:
        # Lấy danh sách tất cả các mục trong thư mục gốc
        directory_contents = os.listdir(base_path)
    except FileNotFoundError:
        print(f"Lỗi: Không tìm thấy đường dẫn '{base_path}'. Vui lòng kiểm tra lại.")
        return
    except Exception as e:
        print(f"Đã xảy ra lỗi khi truy cập đường dẫn: {e}")
        return

    print(f"Bắt đầu đổi tên các thư mục trong: {base_path}")
    print("-" * 50)
    
    # Lặp qua từng mục
    for original_name in directory_contents:
        # Xây dựng đường dẫn đầy đủ
        original_full_path = os.path.join(base_path, original_name)
        
        # Chỉ xử lý các thư mục
        if os.path.isdir(original_full_path):
            
            # Chuẩn hóa tên mới
            new_name = standardize_name(original_name)
            
            # Kiểm tra nếu tên cần được đổi
            if original_name != new_name:
                new_full_path = os.path.join(base_path, new_name)
                
                try:
                    # Thực hiện đổi tên
                    os.rename(original_full_path, new_full_path)
                    print(f"Đã đổi tên: '{original_name}' -> '{new_name}'")
                except Exception as e:
                    print(f"Lỗi khi đổi tên thư mục '{original_name}': {e}")
            else:
                # Trường hợp tên đã chuẩn hóa rồi
                print(f"Không cần đổi tên: '{original_name}'")

    print("-" * 50)
    print("Hoàn thành quá trình đổi tên thư mục.")

# Đường dẫn thư mục gốc mà bạn muốn đổi tên
BASE_DIR = "C:/Users/tam/Desktop/Data/leaf/plantvillage dataset"

# Chạy hàm đổi tên
rename_directories(BASE_DIR)