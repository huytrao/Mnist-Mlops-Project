Đây là project thử nghiệm Mlops.
MLOps là viết tắt của Machine Learning Operations (Hoạt động học máy). MLOps tập trung vào việc hợp lý hóa quy trình triển khai các mô hình học máy vào sản xuất, sau đó duy trì và giám sát chúng. MLOps là một chức năng hợp tác, thường bao gồm các nhà khoa học dữ liệu, kỹ sư ML và kỹ sư DevOps.

### Phần 1: Quản lí các task, data chạy.
Sẽ có 3 task chủ yếu là.
![image](https://github.com/user-attachments/assets/c8e367a5-9b95-42d2-b38a-5b138ec2a4b9)
Trong clear ml nó có phần quản lý dữ liệu, dữ liệu minst sau khi được lọc sẽ được upload lên. và sau đó prepocess và mọi thứ được quản lí.
![image](https://github.com/user-attachments/assets/372170ba-1193-4445-b21b-6fd3901eece2)  Ví dụ database.
Và trainning model 
Script Huấn luyện: Quá trình huấn luyện được thực hiện thông qua script huấn luyện. Script này có thể bao gồm các bước như:
  -Tải dữ liệu: Tải bộ dữ liệu đã được chuẩn bị và tiền xử lý.
  -Xây dựng mô hình: Chọn kiến trúc mô hình phù hợp (ví dụ: mạng nơ-ron tích chập - CNN cho bài toán hình ảnh).
  -Cấu hình huấn luyện: Thiết lập các siêu tham số (hyperparameters) như learning rate, batch size, số epochs, hàm mất mát (loss function), optimizer, v.v.
  -Huấn luyện mô hình: Thực hiện quá trình huấn luyện mô hình trên dữ liệu huấn luyện.
  -Đánh giá mô hình: Đánh giá hiệu suất mô hình trên tập dữ liệu kiểm thử (test dataset) sử dụng các metrics độ chính xác.
  -Lưu trữ mô hình: Lưu trữ mô hình đã huấn luyện cùng với các thông tin liên quan (tham số, metrics, phiên bản dữ liệu) để sử dụng cho các bước tiếp theo.
  -Quản lý Thử nghiệm (Experiment Management).
  ![image](https://github.com/user-attachments/assets/b17afeaf-b707-406a-bb49-3b26afa15ed8)

  -Theo dõi Tham số và Metrics: Trong quá trình huấn luyện, MLOps giúp tự động theo dõi và ghi lại tất cả các tham số huấn luyện (hyperparameters) và các metrics hiệu suất (ví dụ: loss, accuracy) theo thời gian thực. Điều này cho phép bạn giám sát tiến trình huấn luyện và so sánh các thử nghiệm khác nhau.
  -Ghi lại Artifacts: MLOps cũng giúp ghi lại các artifacts quan trọng của quá trình huấn luyện, bao gồm mô hình đã huấn luyện, bộ dữ liệu phiên bản, cấu hình huấn luyện, và các kết quả đánh giá. Điều này đảm bảo rằng tất cả các thành phần của thử nghiệm được lưu trữ và quản lý tập trung.
  ![image](https://github.com/user-attachments/assets/18dd48b4-6d5c-4c22-b7f2-edcca6def3e8)
  


### Phần 2: Tiến hành Xây dựng pipeline để train model.
  Bằng cách cài đặt file pipeline ta có thể chạy một pipelien model lưu lại model tốt nhất và xem.
  Pipeline huấn luyện mô hình là một chuỗi các bước tự động hóa được kết nối với nhau để thực hiện toàn bộ quá trình huấn luyện mô hình machine learning. Mục tiêu là biến đổi dữ liệu thô, chuẩn bị dữ liệu, huấn luyện mô hình, đánh giá hiệu suất, và cuối cùng tạo ra một mô hình đã được huấn luyện và sẵn sàng để triển khai.
  ![image](https://github.com/user-attachments/assets/ab08ce68-0fc1-45e4-a334-16c8cbbf0346)

Sau khi chạy pipeline ta sẽ có best model sau đó ta triển khai 

### Phần 3: Preprocessing within the Serving Pipeline:
Đang làm tới bước
thì trong file preprocess_serving.py
Tiền xử lý dữ liệu đầu vào (dạng dictionary) thành định dạng xgb.DMatrix cho mô hình XGBoost nhận dạng chữ số MNIST.
Hậu xử lý kết quả dự đoán từ mô hình XGBoost, làm tròn và trả về nhãn lớp dự đoán.
Sau đó đưa lên clearml
![image](https://github.com/user-attachments/assets/d84443fa-1231-48b3-a77d-368b18f4ead7)
![image](https://github.com/user-attachments/assets/a74f7684-9c47-408b-9f54-42ff1b6d0c5c)
Còn tiếp.
### Phần 4 Model driff and Monitoring
Còn tiếp.



