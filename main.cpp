#include <openvino/openvino.hpp>
#include<windows.h>
#include<opencv2/opencv.hpp>
using namespace std;
using namespace ov;
using namespace cv;

int main()
{
	//参数定义
	float confidence_threshold = 0.5;	//置信度
	float iou = 0.25;		

	//创建引擎实例
	ov::Core ie;

	//查看设备
	vector<string>devices = ie.get_available_devices();
	for (int i = 0; i < devices.size(); i++)
	{
		printf("device: %s\n", devices[i].c_str());	//GAN是核显
	}

	//加载onnx模型
	//string onnx_path = "C:/Users/Zzzz/Desktop/Z/cf_3000img.onnx";
	string xml_path = "cf_v6_openvino_model/cf_v6.xml";
	string bin_path = "cf_v6_openvino_model/cf_v6.bin";
	CompiledModel model = ie.compile_model(xml_path, "AUTO");	//加载模型,以cpu推理,cpu or cuda,else auto


	InferRequest net = model.create_infer_request();	//实例化

	//获取网络输入		[b,c,w,h]->对应[输入图片数量, 通道数, 宽, 高]
	Tensor input_tensor = net.get_input_tensor();	//获取网络的输入信息
	Shape input_shape = input_tensor.get_shape();	//获取输入信息的维度
	//size_t c = input_shape[0];	//输入图片数量	//模型为1,不对这个维度感兴趣
	size_t channels = input_shape[1];	//通道数		
	size_t w = input_shape[2];			//网络输入w要求
	size_t h = input_shape[3];
	size_t image_size = h * w;		//输入图像尺寸	

	//读取图片
	//Mat img = cv::imread("C:/Users/Zzzz/Desktop/Z/000102.png");


	int Desktop_W, Desktop_H, Desktop_x, Desktop_y,BitBlt_x,BitBlt_y, BitBlt_W, BitBlt_H;
	//显示器分辨率
	Desktop_W = 1920;	//水平长度
	Desktop_H = 1080;	//垂直长度
	//中心点
	Desktop_x = Desktop_W / 2;
	Desktop_y = Desktop_H / 2;

	//截图宽高
	BitBlt_W = Desktop_W * 0.5;
	BitBlt_H = Desktop_H * 0.5;

	//截图原点
	BitBlt_x = Desktop_x - (BitBlt_W/2);
	BitBlt_y = Desktop_y - (BitBlt_H/2);
	cout << "Desktop size:"<< Desktop_W << " x "<<Desktop_H << endl	;
	cout << "BitBlt range ->  x:" << BitBlt_x << " y:" << BitBlt_y << " width:" << BitBlt_W << " height:" << BitBlt_H << endl;
	HBITMAP	BitMap;
	//1.获取屏幕句柄
	HWND hwnd = GetDesktopWindow();
	//2.获取屏幕DC
	HDC hdc = GetWindowDC(hwnd);
	//3.创建兼容DC(内存DC)
	HDC	mfdc = CreateCompatibleDC(hdc);
	//5.创建位图Bitmap对象
	BitMap = CreateCompatibleBitmap(hdc, BitBlt_W, BitBlt_H);
	//6.将位图对象放入内存dc(也可以说是绑定)
	SelectObject(mfdc, BitMap);

	Mat scr,img;
	//7.创建一个维度和截图宽高的空位图,
	scr.create(Size(BitBlt_W, BitBlt_H), CV_8UC4);
	namedWindow("img", WINDOW_NORMAL);	//创建窗口
	while (true)
	{

		double t1 = (double)getTickCount();
		if (getWindowProperty("img", WND_PROP_VISIBLE) != 1)
		{
			break;
		}
		BitBlt(mfdc, 0, 0, BitBlt_W, BitBlt_H, hdc, BitBlt_x, BitBlt_y, SRCCOPY);
		GetBitmapBits(BitMap, BitBlt_W * BitBlt_H * 4, scr.data);		//将BitBlt的位图信息传入
		cvtColor(scr, img, COLOR_BGRA2BGR);

		//预处理
		Mat blob;	//创建一个空矩阵
		cv::resize(img, blob, cv::Size(h, w));	//裁剪读到的图片赋值给空矩阵blob
		//cv::cvtColor(blob_image, blob_image, cv::COLOR_BGR2RGB);	//RB转换,BGR转为RGB
		blob.convertTo(blob, CV_32F);	//整数转为浮点,网络要求是浮点
		blob = blob / 255.0;	//归一化
		float x_factor = img.cols / 640.0f;     //宽的比例,用于还原坐标
		float y_factor = img.rows / 640.0f;     //高的比例

		//NCHW bcwh ,将blob数据按位传入input_data
		float* input_data = input_tensor.data<float>();//由指针指向每个像素
		for (size_t row = 0; row < h; row++)
		{
			for (size_t col = 0; col < w; col++)
			{
				for (size_t c = 0; c < channels; c++)
				{
					input_data[image_size * c + row * w + col] = blob.at<Vec3f>(row, col)[c];
				}
			}
		}

		//推理
		net.infer();

		//获取输出结果
		auto output = net.get_tensor("output");	//获取输出结果
		float* output_blob = (float*)output.data();	//创建指向输出矩阵的指针,用以下面的拷贝

		//获取输出维度
		ov::Shape output_shape = output.get_shape();	//[1,25200,7]
		size_t outs_rows = output_shape[1];	//25200		//这个是先验框的个数
		size_t outs_cols = output_shape[2];	// 7  : x y w h 1 c0 c1 

		//创建存储容器
		vector<cv::Rect> boxes;         //存储坐标  //循环清空
		//vector<int> classIds;           //存储类别
		vector<float> confidences;      //存储置信度
		Mat det_output = Mat(outs_rows, outs_cols, CV_32FC1, output_blob);	//从output_blob拷贝一个outs_rows行,outs_cols列,CV_32FC1通道的Mat矩阵,output_blob传入指针

		for (int i = 0; i < outs_rows; i++)		//遍历所有的先验框
		{
			//获取置信度
			float conf = det_output.at<float>(i, 4);	//获取第i维度(第i行)下的第4个值,每一个i维度都是[x y w h 1 c0 c1 ]	
			if (conf < confidence_threshold)         //筛选置信度阈值的
			{
				continue;
			}
			//筛选置信度,用于iou计算
			cv::Mat classes_scores = det_output.row(i).colRange(5, outs_cols);   //colRange为指定的列范围创建矩阵标头。
																								//即按det_output的第i行的第5为和第outs_cols范围创建索引,具体百度Mat colRange
			cv::Point classIdPoint;         //存储得到的最大值的对应像素坐标
			double score;           //存储最大值的变量
			//找到最大的置信度
			minMaxLoc(classes_scores, 0, &score, 0, &classIdPoint); //寻找classes_scores(一维数组当作向量) 中最大值.放入classIdPoint

			//坐标提取
			float cx = det_output.at<float>(i, 0);
			float cy = det_output.at<float>(i, 1);
			float ow = det_output.at<float>(i, 2);
			float oh = det_output.at<float>(i, 3);

			//转为原图坐标
			int x = static_cast<int>((cx - 0.5 * ow) * x_factor);
			int y = static_cast<int>((cy - 0.5 * oh) * y_factor);
			int width = static_cast<int>(ow * x_factor);
			int height = static_cast<int>(oh * y_factor);

			cv::Rect box;
			box.x = x;
			box.y = y;
			box.width = width;
			box.height = height;
			boxes.push_back(box);   //将xywh作为一个元素添加进去
			//classIds.push_back(classIdPoint.x); //类别
			confidences.push_back(score);   //置信度
		}

		//iou
		std::vector<int> indexes;       //创建一个索引数组
		dnn::NMSBoxes(boxes, confidences, confidence_threshold, iou, indexes);  //非极大值抑制计算,然后返回一个索引数组,每个索引对应一个框

		for (size_t i = 0; i < indexes.size(); i++)     //遍历索引
		{
			int index = indexes[i];
			//cout << "!!!!!!!!" << boxes[index] << endl;

			cv::rectangle(img, boxes[index], cv::Scalar(0, 255, 0), 2, 8);
		}	

		//画fps,time
		ostringstream ss;
		vector<double> layersTimings;
		//double freq = cv::getTickFrequency() / 1000.0;

		double time = (getTickCount() - t1) * 1000 / (getTickFrequency());
		ss << "FPS: " << fixed << setprecision(2) << 1000 / time << " ;TIME: " << time << "ms";//fixed << setprecision(3);控制后面输出的小数点位
		putText(img, ss.str(), cv::Point(20, 40), cv::FONT_HERSHEY_PLAIN, 2.0, cv::Scalar(255, 255, 0), 2, 8);

		//显示
		//cout << "输出:" << outs_rows << " x " << outs_cols << "\n处理时间:" << time << " s" << std::endl; //2改3.
		//resize(img, img, Size(BitBlt_W, BitBlt_H));
		imshow("img", img);
		waitKey(1);
	}

	system("pause");
	return 0;
}

