import json  
import struct  

# Hàm để lưu mô hình vào file nhị phân  
def save_model_to_binary(model_json, path):  
    with open(path, 'wb') as bin_file:  
        num_trees = len(model_json['models'])  
        bin_file.write(struct.pack('I', num_trees))  

        for tree in model_json['models']:  
            save_tree_to_binary(bin_file, tree)  

def save_tree_to_binary(bin_file, node):  
    if isinstance(node, dict):  
        # Ghi đánh dấu cho nút bình thường (0)  
        bin_file.write(struct.pack('B', 0))  # 0: không phải leaf  

        # Ghi thông tin cho nút bình thường  
        bin_file.write(struct.pack('I', node['fid']))  # Ghi fid  
        # bin_file.write(struct.pack('d', node['split_point']))  # Ghi split_point 
        
        bin_file.write(struct.pack('d', round(node['split_point'], 2)))  # Ghi split_point làm tròn 2 chữ số  
 
        # bin_file.write(struct.pack('d', node['gain']))  # Ghi gain
         
        bin_file.write(struct.pack('d', round(node['gain'], 2)))  # Ghi gain làm tròn 2 chữ số  
 

        # Ghi cây con trái và phải  
        save_tree_to_binary(bin_file, node['left'])  
        save_tree_to_binary(bin_file, node['right'])  
    else:  
        # Ghi đánh dấu cho nút lá (1)  
        bin_file.write(struct.pack('B', 1))  # 1: là leaf  
        # bin_file.write(struct.pack('d', node))  # Ghi giá trị lá  
        
        bin_file.write(struct.pack('d', round(node, 2)))  # Ghi giá trị lá làm tròn 2 chữ số  


# Hàm để đọc mô hình từ file nhị phân  
def load_model_from_binary(filepath):  
    with open(filepath, 'rb') as bin_file:  
        num_trees_bytes = bin_file.read(4)  
        if not num_trees_bytes:  
            raise ValueError("File nhị phân rỗng hoặc không hợp lệ.")  
        num_trees = struct.unpack('I', num_trees_bytes)[0]  

        models = []  
        for _ in range(num_trees):  
            models.append(load_tree_from_binary(bin_file))  

    return {  
        'n_estimators': num_trees,  
        'models': models  
    }  

def load_tree_from_binary(bin_file):  
    # Đọc đánh dấu để xác định loại nút  
    flag_byte = bin_file.read(1)  
    if not flag_byte:  
        return None  # Không còn dữ liệu, trả về None  

    is_leaf = struct.unpack('B', flag_byte)[0]  # 0: không phải leaf, 1: là leaf  

    if is_leaf == 1:  
        # Nếu là nút lá, đọc giá trị lá  
        leaf_value_bytes = bin_file.read(8)  # Đọc 8 byte cho giá trị lá  
        return struct.unpack('d', leaf_value_bytes)[0]  # Trả về giá trị lá  

    # Nếu là nút bình thường, đọc thông tin khác  
    node = {}  
    node['fid'] = struct.unpack('I', bin_file.read(4))[0]  # Đọc fid  
    node['split_point'] = struct.unpack('d', bin_file.read(8))[0]  # Đọc split_point  
    node['gain'] = struct.unpack('d', bin_file.read(8))[0]  # Đọc gain  

    # Đệ quy đọc cây con trái và phải  
    node['left'] = load_tree_from_binary(bin_file)  
    node['right'] = load_tree_from_binary(bin_file)  

    return node  

# Đọc từ file JSON  
file_json_path = r'C:\Users\ASUS\Documents\BTL_AI\AI_BTL\models\v1\model_2.json'  # Đường dẫn tới file JSON của bạn  
file_bst_path = r'C:\Users\ASUS\Documents\BTL_AI\AI_BTL\models\v1\model_2.bst'  # Đường dẫn tới file nhị phân muốn lưu  

# Lưu mô hình sang định dạng nhị phân  
with open(file_json_path, 'r') as json_file:  
    model_json = json.load(json_file)  

save_model_to_binary(model_json, file_bst_path)  

# Đọc mô hình từ file nhị phân  
loaded_model = load_model_from_binary(file_bst_path)  

# Hiển thị thông tin cây để xác nhận  
print("Mô hình được tải thành công:")  
print(json.dumps(loaded_model, indent=4))