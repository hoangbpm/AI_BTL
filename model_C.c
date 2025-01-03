#include <stdio.h>  
#include <stdlib.h>  
#include <string.h>  
#include <math.h>  // Thêm thư viện này để sử dụng hàm exp  

// Định nghĩa cấu trúc cho nút của cây  
typedef struct TreeNode {  
    unsigned int feature_id;     // ID thuộc tính để phân chia  
    double split_point;          // Điểm phân chia  
    double gain;                 // Giá trị gain  
    struct TreeNode* left;       // Con trỏ đến nút con trái  
    struct TreeNode* right;      // Con trỏ đến nút con phải  
} TreeNode;  

// Hàm để đọc cây từ tệp nhị phân  
TreeNode* loadTreeFromBinary(FILE* bin_file) {  
    unsigned char flag;  
    if (fread(&flag, sizeof(flag), 1, bin_file) != 1) {  
        return NULL; // Không thể đọc, trả về NULL  
    }  

    TreeNode* node = (TreeNode*)malloc(sizeof(TreeNode));  
    if (node == NULL) {  
        return NULL; // Kiểm tra cấp phát bộ nhớ  
    }  

    if (flag == 1) { // Nút lá  
        double leaf_value;  
        if (fread(&leaf_value, sizeof(leaf_value), 1, bin_file) != 1) {  
            free(node);  
            return NULL; // Không thể đọc giá trị lá  
        }  
        node->feature_id = 0;       // ID không cần thiết cho lá  
        node->split_point = leaf_value;  
        node->gain = 0;  
        node->left = NULL;  
        node->right = NULL;  
    } else { // Nút bình thường  
        if (fread(&node->feature_id, sizeof(node->feature_id), 1, bin_file) != 1 ||  
            fread(&node->split_point, sizeof(node->split_point), 1, bin_file) != 1 ||  
            fread(&node->gain, sizeof(node->gain), 1, bin_file) != 1) {  
            free(node);  
            return NULL; // Không thể đọc dữ liệu nút  
        }  

        // Đọc đệ quy các nút con  
        node->left = loadTreeFromBinary(bin_file);  
        node->right = loadTreeFromBinary(bin_file);  
    }  

    return node; // Trả về nút đã đọc  
}  

// Định nghĩa cấu trúc cho cây phân loại  
typedef struct MyXGBClassificationTree {  
    int max_depth;             // Độ sâu tối đa  
    double reg_lambda;         // Hệ số điều chỉnh  
    double prune_gamma;        // Ngưỡng cắt tỉa  
    TreeNode* root;            // Nút gốc  
} MyXGBClassificationTree;  

// Hàm khởi tạo cây phân loại  
MyXGBClassificationTree* createMyXGBClassificationTree(TreeNode* root, int max_depth, double reg_lambda, double prune_gamma) {  
    MyXGBClassificationTree* tree = (MyXGBClassificationTree*)malloc(sizeof(MyXGBClassificationTree));  
    if (tree == NULL) {  
        return NULL; // Kiểm tra cấp phát bộ nhớ  
    }  
    tree->max_depth = max_depth;  
    tree->reg_lambda = reg_lambda;  
    tree->prune_gamma = prune_gamma;  
    tree->root = root;  
    return tree;  
}  
// Hàm để giải phóng bộ nhớ cho cây  
void freeTree(TreeNode* node) {  
    if (node == NULL) return; // Nếu nút là NULL thì không làm gì  
    freeTree(node->left);     // Giải phóng cây con trái  
    freeTree(node->right);    // Giải phóng cây con phải  
    free(node);               // Giải phóng nút hiện tại  
}  


// Hàm để đọc mô hình từ tệp nhị phân, trả về MyXGBClassificationTree**  
MyXGBClassificationTree** loadModelFromBinary(const char* filepath, unsigned int* num_trees) {  
    FILE* bin_file = fopen(filepath, "rb");  
    if (!bin_file) {  
        perror("Cannot open binary file");  
        return NULL;  
    }  

    // Đọc số lượng cây  
    if (fread(num_trees, sizeof(unsigned int), 1, bin_file) != 1) {  
        fclose(bin_file);  
        perror("Cannot read the number of trees");  
        return NULL;  
    }  

    // Tạo mảng để lưu cây  
    MyXGBClassificationTree** trees = (MyXGBClassificationTree**)malloc((*num_trees) * sizeof(MyXGBClassificationTree*));  
    if (trees == NULL) {  
        fclose(bin_file);  
        return NULL; // Kiểm tra cấp phát bộ nhớ  
    }  

    for (unsigned int i = 0; i < *num_trees; ++i) {  
        TreeNode* root = loadTreeFromBinary(bin_file); // Tải cây từ tệp nhị phân  
        if (root == NULL) {  
            for (unsigned int j = 0; j < i; ++j) {  
                freeTree(trees[j]->root);  
                free(trees[j]);  
            }  
            free(trees);  
            fclose(bin_file);  
            return NULL; // Nếu không đọc được cây, giải phóng bộ nhớ  
        }  
        trees[i] = createMyXGBClassificationTree(root, 0, 0.0, 0.0); // Khởi tạo cây phân loại  
    }  

    fclose(bin_file);  
    return trees; // Trả về mảng chứa các cây  
}  


// Hàm xuất thông tin của cây để kiểm tra  
void printTree(TreeNode* node, int depth) {  
    if (node == NULL) {  
        printf("%*sNULL\n", depth * 2, ""); // In NULL nếu nút là NULL  
        return;  
    }  

    if (node->left == NULL && node->right == NULL) {  
        printf("%*sLeaf: %f\n", depth * 2, "", node->split_point); // In giá trị lá  
    } else {  
        printf("%*sNode: feature_id=%u, split_point=%f, gain=%f\n", depth * 2, "",  
               node->feature_id, node->split_point, node->gain);  
        printTree(node->left, depth + 1);  // In thông tin cây con trái  
        printTree(node->right, depth + 1); // In thông tin cây con phải  
    }  
}  

// Hàm dự đoán cho cây  
double predictRecursively(TreeNode* node, const double* x) {  
    if (node == NULL) {  
        return 0.0; // Nếu là NULL, trả về 0.0  
    }  

    if (node->left == NULL && node->right == NULL) {  
        return node->split_point; // Trả giá trị của nút lá  
    }  

    if (x[node->feature_id] <= node->split_point) {  
        return predictRecursively(node->left, x); // Dự đoán ở cây con trái  
    } else {  
        return predictRecursively(node->right, x); // Dự đoán ở cây con phải  
    }  
}  

// Hàm dự đoán cho tất cả các mẫu  
double* predict(MyXGBClassificationTree* tree, const double** x_test, unsigned int num_samples) {  
    double* predictions = (double*)malloc(num_samples * sizeof(double)); // Mảng lưu dự đoán  
    if (predictions == NULL) return NULL; // Kiểm tra cấp phát bộ nhớ  

    for (unsigned int i = 0; i < num_samples; ++i) {  
        predictions[i] = predictRecursively(tree->root, x_test[i]); // Gọi hàm dự đoán cho mỗi mẫu  
    }  
    return predictions; // Trả về mảng dự đoán  
}  

// Cấu trúc cho bộ phân loại nhiều cây  
typedef struct MyXGBClassifier {  
    MyXGBClassificationTree** models; // Mảng chứa các cây  
    unsigned int num_models;           // Số lượng mô hình  
    double learning_rate;              // Tỷ lệ học  
    double base_score; 
                    // Điểm cơ sở  
} MyXGBClassifier;  

// Hàm khởi tạo bộ phân loại  
MyXGBClassifier* createMyXGBClassifier() {  
    MyXGBClassifier* classifier = (MyXGBClassifier*)malloc(sizeof(MyXGBClassifier));  
    if (classifier == NULL) {  
        perror("Failed to allocate memory for classifier");  
        return NULL;  
    }  
    classifier->models = NULL; // Chưa có cây nào  
    classifier->num_models = 0;  
    classifier->learning_rate = 0.3;  // Tỷ lệ học mặc định  
    classifier->base_score = 0.5;     // Điểm cơ sở mặc định  
    return classifier;  
}  

// Hàm để giải phóng bộ nhớ cho bộ phân loại  
void freeMyXGBClassifier(MyXGBClassifier* classifier) {  
    if (classifier != NULL) {  
        for (unsigned int i = 0; i < classifier->num_models; ++i) {  
            freeTree(classifier->models[i]->root); // Giải phóng nút gốc trong cây  
            free(classifier->models[i]);       // Giải phóng cây  
        }  
        free(classifier->models); // Giải phóng mảng cây  
        free(classifier);         // Giải phóng bộ phân loại  
    }  
}  

// Hàm để tải các mô hình từ tệp nhị phân  
void loadModels(MyXGBClassifier* classifier, const char* filepath) {  
    unsigned int num_trees;  
    classifier->models = loadModelFromBinary(filepath, &num_trees);  

    // Kiểm tra xem việc tải cây có thành công không  
    if (classifier->models == NULL) {  
        fprintf(stderr, "Failed to load models from file: %s\n", filepath);  
        return;  
    }  

    classifier->num_models = num_trees; // Cập nhật số lượng mô hình  
    // printf("Finished loading %u models from trees.\n", num_trees); // In thông báo nếu cần  
}  

// Hàm dự đoán cho lớp phân loại  
double* classifierPredict(MyXGBClassifier* classifier, const double** x_test, unsigned int num_samples) {  
    double* Fm = (double*)malloc(num_samples * sizeof(double)); // Mảng lưu trữ giá trị logits  
    if (Fm == NULL) {  
        perror("Failed to allocate memory for predictions");  
        return NULL; // Xử lý lỗi nếu không đủ bộ nhớ  
    }  

    // Khởi tạo Fm với base_score  
    for (unsigned int i = 0; i < num_samples; ++i) {  
        Fm[i] = classifier->base_score;  
    }  

    // Dự đoán từ từng cây  
    for (unsigned int m = 0; m < classifier->num_models; ++m) {  
        double* model_predictions = predict(classifier->models[m], x_test, num_samples); // Dự đoán từ cây  
        if (model_predictions == NULL) {  
            free(Fm);  
            return NULL; // Kiểm tra lỗi  
        }  
        for (unsigned int i = 0; i < num_samples; ++i) {  
            Fm[i] += classifier->learning_rate * model_predictions[i]; // Cập nhật giá trị logits  
        }  
        free(model_predictions); // Giải phóng mảng dự đoán từ cây sau khi sử dụng  
    }  

    // In giá trị logits  
    printf("Predictions before probability conversion:\n");  
    for (unsigned int i = 0; i < num_samples; ++i) {  
        printf("%.4f ", Fm[i]); // In giá trị logits  
    }  
    printf("\n");  

    // Chuyển đổi logits thành xác suất  
    double* probabilities = (double*)malloc(num_samples * sizeof(double));  
    if (probabilities == NULL) {  
        free(Fm);  
        return NULL; // Kiểm tra lỗi  
    }  

    printf("Predicted probabilities:\n");  
    for (unsigned int i = 0; i < num_samples; ++i) {  
        probabilities[i] = 1.0 / (1.0 + exp(-Fm[i])); // Thực hiện chuyển đổi sigmoid  
        printf("%.4f ", probabilities[i]); // In xác suất  
    }  
    printf("\n");  

    free(Fm); // Giải phóng bộ nhớ cho mảng logits  
    return probabilities; // Trả về mảng xác suất  
}  

// Hàm chính  


int main() {  
    printf("PREDICT V4\n");  
    MyXGBClassifier* classifier = createMyXGBClassifier(); // Tạo classifier  
    if (classifier == NULL) {  
        return 1; // Thoát nếu không thể tạo classifier  
    }  

    // Mảng tên tệp mô hình (7 mô hình)  
    const char* model_files[] = {  
        "C:\\Users\\ASUS\\Documents\\BTL_AI\\AI_BTL\\models\\v5\\model_1.bst",  
        "C:\\Users\\ASUS\\Documents\\BTL_AI\\AI_BTL\\models\\v5\\model_2.bst",  
        "C:\\Users\\ASUS\\Documents\\BTL_AI\\AI_BTL\\models\\v5\\model_3.bst",  
        "C:\\Users\\ASUS\\Documents\\BTL_AI\\AI_BTL\\models\\v5\\model_4.bst",  
        "C:\\Users\\ASUS\\Documents\\BTL_AI\\AI_BTL\\models\\v5\\model_5.bst",  
        "C:\\Users\\ASUS\\Documents\\BTL_AI\\AI_BTL\\models\\v5\\model_6.bst",  
        "C:\\Users\\ASUS\\Documents\\BTL_AI\\AI_BTL\\models\\v5\\model_7.bst",  
    };  

    // Số lượng mô hình và mẫu  
    const unsigned int num_models = sizeof(model_files) / sizeof(model_files[0]);  
    const double* x_test[] = {  
        (double[]){78.46, 87.51, 24.49, 151.35},  
        (double[]){77.94, 58.17, 18.31, 73.59},
        (double[]){69.09, 51.7, 21.24, 616.62} , 
        (double[]){31.78, 24.39, 11.0, 87.08}  ,
        (double[]){40.47, 59.21, 12.13, 918.97}  ,
        (double[]){50.96, 26.00, 26.81, 792.59}  ,
        (double[]){17.21, 27.91, 32.61, 277.96}  
 
    };  

    unsigned int num_samples = sizeof(x_test) / sizeof(x_test[0]); // Tính số lượng mẫu  

    // Mảng để lưu trữ dự đoán cho mỗi mẫu qua tất cả các mô hình  
    int predictions_arr[num_samples][num_models];  

    // Tải mô hình từng cái và dự đoán sử dụng từng cái  
    for (unsigned int i = 0; i < num_models; ++i) {  
        loadModels(classifier, model_files[i]); // Tải từng mô hình  

        // Tạo dự đoán cho mô hình hiện tại  
        double* predictions = classifierPredict(classifier, x_test, num_samples);  
        
        // Xuất dự đoán cho mô hình hiện tại  
        if (predictions != NULL) {  
            printf("Predictions for Model %u:\n", i + 1);  // Đếm từ 1 để dễ hiểu  
            for (unsigned int j = 0; j < num_samples; ++j) {  
                // Áp dụng ngưỡng và lưu trữ dưới dạng số nguyên (0 hoặc 1)  
                predictions_arr[j][i] = (predictions[j] >= 0.5) ? 1 : 0;  
                printf("Sample %u: %d\n", j, predictions_arr[j][i]); // In dự đoán nhị phân  
            }  
            free(predictions); // Giải phóng bộ nhớ cho mảng dự đoán  
        } else {  
            printf("No predictions available for Model %u.\n", i + 1); // Xử lý trường hợp không có dự đoán  
        }  
    }  

    // Xuất mảng dự đoán đầy đủ cho tất cả các mẫu  
    printf("Final Predictions Array:\n");  
    for (unsigned int j = 0; j < num_samples; ++j) {  
        printf("Sample %u Predictions: [", j);  
        for (unsigned int k = 0; k < num_models; ++k) {  
            printf("%d", predictions_arr[j][k]);  
            if (k < num_models - 1) printf(", "); // In dấu phẩy giữa các phần tử  
        }  
        printf("]\n");  
    }  

    // Xuất kết quả cuối cùng dựa trên dự đoán của mô hình  
    printf("Final Interpreted Results:\n");  
    for (unsigned int j = 0; j < num_samples; ++j) {  
        int interpreted_result = 0;  
        int combination = 0;  

        // Tính toán sự kết hợp dựa trên các dự đoán  
        for (unsigned int k = 0; k < num_models; ++k) {  
            combination += predictions_arr[j][k] * (1 << k); // Sử dụng bit shift để tạo số nhị phân  
        }  

        // Giải thích kết quả dựa trên sự kết hợp cụ thể  
        switch (combination) {  
            case 0b0000001: interpreted_result = 1; break; // Chỉ model 1  
            case 0b0000010: interpreted_result = 2; break; // Chỉ model 2  
            case 0b0000100: interpreted_result = 3; break; // Chỉ model 3  
            case 0b0001000: interpreted_result = 4; break; // Chỉ model 4  
            case 0b0010000: interpreted_result = 5; break; // Chỉ model 5  
            case 0b0100000: interpreted_result = 6; break; // Chỉ model 6  
            case 0b1000000: interpreted_result = 7; break; // Chỉ model 7  
            default: interpreted_result = 1; break; // Không có mô hình nào  
        }  

        printf("Sample %u Interpreted Result: %d\n", j, interpreted_result);  
    }  

    // Giải phóng bộ nhớ cho classifier  
    freeMyXGBClassifier(classifier);  

    return 0; // Trả về mã 0 để chỉ ra thành công  
}  
