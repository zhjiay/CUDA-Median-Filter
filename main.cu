#include <iostream>
#include <opencv2/opencv.hpp>
#include <numeric>
#include<vector>
#include <string>
#include <chrono>

#include<cuda.h>
#include<cuda_runtime.h>

class Timer{
public:
    Timer(){}
    ~Timer(){}
    void begin(){
        _start=std::chrono::high_resolution_clock::now();
    }
    double end(){
        auto _end=std::chrono::high_resolution_clock::now();
        double res=std::chrono::duration_cast<std::chrono::microseconds>(_end-_start).count()/1000.0;// 返回 ms ;
        return res;
    }
private:
    std::chrono::high_resolution_clock::time_point _start;
};


__device__
float cudaMedianValue(float* frr, int Len,int half){
    // median value find
    for(int i=0; i<half+1; i++){
        float temp_min=frr[i];
        int temp_idx=i;
        for(int j=i+1; j<Len; j++){
            if(frr[j]<temp_min) {
                temp_min=frr[j];
                temp_idx=j;
            }
        }
        if(temp_idx!=i){
            frr[temp_idx]=frr[i];
            frr[i]=temp_min;
        }
    }
    return frr[half];
}

__device__
uchar cudaMedianValue(uchar* frr, int Len,int half){
    // median value find
    for(int i=0; i<half+1; i++){
        uchar temp_min=frr[i];
        int temp_idx=i;
        for(int j=i+1; j<Len; j++){
            if(frr[j]<temp_min) {
                temp_min=frr[j];
                temp_idx=j;
            }
        }
        if(temp_idx!=i){
            frr[temp_idx]=frr[i];
            frr[i]=temp_min;
        }
    }
    //bubble sort
//    for(int i=0; i<Len; i++){
//        for(int j=0; j<Len-i-1; j++){
//            if( frr[j]>frr[j+1]){
//                uchar temp=frr[j];
//                frr[j]=frr[j+1];
//                frr[j+1]=temp;
//            }
//        }
//    }

    return frr[half];
}

__global__
void cudaMedianFilter7(float* d_img, float* d_res, int rows, int cols){
    int r=blockIdx.x*blockDim.x + threadIdx.x;
    int c=blockIdx.y*blockDim.y + threadIdx.y;

    if(r<0 || r>=rows || c<0 ||c>=cols)  return;

    const int filterSize=7;
    float frr[filterSize*filterSize];

    int halfLen=filterSize/2;
    for(int rr=-halfLen; rr<=halfLen; rr++){
        for(int cc=-halfLen; cc<=halfLen; cc++){
            int trr=r+rr;
            int tcc=c+cc;
            if(trr<0) trr=0;
            if(trr>=rows) trr=rows-1;
            if(tcc<0) tcc=0;
            if(tcc>=cols) tcc=cols-1;
            frr[(rr+halfLen)*filterSize+cc+halfLen]=d_img[trr*cols+tcc];
        }
    }
    d_res[r*cols+c]=cudaMedianValue(frr, filterSize*filterSize, filterSize*filterSize/2);
}

__global__
void cudaMedianFilter5(float* d_img, float* d_res, int rows, int cols){
    int r=blockIdx.x*blockDim.x + threadIdx.x;
    int c=blockIdx.y*blockDim.y + threadIdx.y;

    if(r<0 || r>=rows || c<0 ||c>=cols)  return;

    const int filterSize=5;
    float frr[filterSize*filterSize];

    int halfLen=filterSize/2;
    for(int rr=-halfLen; rr<=halfLen; rr++){
        for(int cc=-halfLen; cc<=halfLen; cc++){
            int trr=r+rr;
            int tcc=c+cc;
            if(trr<0) trr=0;
            if(trr>=rows) trr=rows-1;
            if(tcc<0) tcc=0;
            if(tcc>=cols) tcc=cols-1;
            frr[(rr+halfLen)*filterSize+cc+halfLen]=d_img[trr*cols+tcc];
        }
    }

    d_res[r*cols+c]=cudaMedianValue(frr, filterSize*filterSize, (filterSize*filterSize)/2);
}

__global__
void cudaMedianFilter3(float* d_img, float* d_res, int rows, int cols){
    int r=blockIdx.x*blockDim.x + threadIdx.x;
    int c=blockIdx.y*blockDim.y + threadIdx.y;

    if(r<0 || r>=rows || c<0 ||c>=cols)  return;

    const int filterSize=3;
    float frr[filterSize*filterSize];
    int halfLen=filterSize/2;
    for(int rr=-halfLen; rr<=halfLen; rr++){
        for(int cc=-halfLen; cc<=halfLen; cc++){
            int trr=r+rr;
            int tcc=c+cc;
            if(trr<0) trr=0;
            if(trr>=rows) trr=rows-1;
            if(tcc<0) tcc=0;
            if(tcc>=cols) tcc=cols-1;
            frr[(rr+halfLen)*filterSize+cc+halfLen]=d_img[trr*cols+tcc];
        }
    }
    d_res[r*cols+c]=cudaMedianValue(frr, filterSize*filterSize, (filterSize*filterSize)/2);
}

__global__
void cudaMedianFilter5(uchar* d_img, uchar* d_res, int rows, int cols){
    int r=blockIdx.x*blockDim.x + threadIdx.x;
    int c=blockIdx.y*blockDim.y + threadIdx.y;
    if(r<0 || r>=rows || c<0 ||c>=cols)  return;

    const int ft=5;
    const int N=ft*ft;
    uchar frr[N]={255};
    const int ht=ft/2;
    for(int rr=-ht; rr<=ht; rr++){
        for(int cc=-ht; cc<=ht; cc++){
            int trr=r+rr;
            int tcc=c+cc;
            if(trr<0) trr=0;
            if(trr>=rows) trr=rows-1;
            if(tcc<0) tcc=0;
            if(tcc>=cols) tcc=cols-1;
            frr[(rr+ht)*ft+cc+ht]=d_img[trr*cols+tcc];
        }
    }
    d_res[r*cols+c]=cudaMedianValue(frr, N, N/2);
}




void test_float_filter();

void test_uchar_filer();

void test0();

int main() {
    test_float_filter();
    return 0;
}

void test_uchar_filer(){

}

void test0(){
    std::string dir_path="C:\\cudacode\\CUDA-Median-Filter\\imgs\\";
    std::string path=dir_path+"org.bmp";
    cv::Mat img=cv::imread(path, 0);
    int rows=img.rows;
    int cols=img.cols;
    int N=rows*cols;

    cv::Mat res0(rows, cols, CV_8UC1, cv::Scalar(0));
    cv::medianBlur(img, res0, 5);
    cv::imwrite(dir_path+"cpufilter.bmp", res0);
    cv::imshow("res_cpu", res0);
    cv::waitKey();

    uchar* d_mat;
    cudaMalloc((void**)&d_mat, N*sizeof(uchar));
    cudaMemcpy(d_mat, img.ptr<uchar>(), N*sizeof(uchar), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();

    uchar* d_res;
    cudaMalloc((void**)&d_res, N*sizeof(uchar));
    cudaMemset(d_res, 128, N*sizeof(uchar));

    dim3 blocksize(32,32,1);
    dim3 gridSize(rows/blocksize.x+1, cols/blocksize.y+1,1);

    cudaMedianFilter5<<<gridSize, blocksize>>>(d_mat, d_res, rows, cols);
    cudaDeviceSynchronize();

    cv::Mat res1(rows, cols, CV_8UC1, cv::Scalar(0));
    cudaMemcpy(res1.ptr<uchar>(), d_res, N*sizeof(uchar), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    cv::imwrite(dir_path+"gpufilter.bmp", res1);
    cv::imshow("res_gpu", res1);
    cv::waitKey();
}


void test_float_filter(){
    int len=999;
    int rows=5120;
    int cols=5120;

    int N=rows*cols;
    Timer tm;
    std::vector<double> cpu_times;
    std::vector<double> gpu_times;
    std::vector<double> diff_res;

    std::vector<double> mem_times;

    dim3 block_size(32,32,1);
    dim3 grid_size(rows/block_size.x+1, cols/block_size.y+1, 1);

    for(int i=0; i<len; i++){
        std::cout<<i<<" ";
        cv::Mat mat(rows, cols, CV_32FC1);
        cv::randu(mat, 0.0f, 255.0f);

        cv::Mat cpu_res(rows, cols, CV_32FC1, cv::Scalar(0.0f));
        tm.begin();
        cv::medianBlur(mat, cpu_res,5); // opencv medianFilter;
        cpu_times.emplace_back(tm.end());

        float* d_mat;
        cudaMalloc((void**)&d_mat,N*sizeof(float));
        tm.begin();
        cudaMemcpy(d_mat, mat.ptr<float>(), N*sizeof(float), cudaMemcpyHostToDevice);
        double mem_t=tm.end();

        float* d_res;
        cudaMalloc((void**)&d_res,N*sizeof(float));

        tm.begin();
        cudaMedianFilter7<<<grid_size, block_size>>>(d_mat, d_res, rows, cols); // cuda medianFilter;
        cudaDeviceSynchronize();
        gpu_times.emplace_back(tm.end());

        cv::Mat cuda_res(rows, cols, CV_32FC1, cv::Scalar(1));
        tm.begin();
        cudaMemcpy(cuda_res.ptr<float>(), d_res, N*sizeof(float), cudaMemcpyDeviceToHost);
        mem_t+=tm.end();
        mem_times.push_back(mem_t);

        cudaFree(d_mat);
        cudaFree(d_res);

        cv::Mat dt=cpu_res-cuda_res;
        cv::Mat abs_dt=cv::abs(dt);
        auto d_sum=cv::sum(abs_dt);
        std::cout<<d_sum[0]<<std::endl;
        diff_res.emplace_back(d_sum[0]);
    }

    for(int i=0;i<len;i++){
        std::cout<<cpu_times[i]<<"  "<<gpu_times[i]<<"  ="<<cpu_times[i]-gpu_times[i]<<std::endl;
    }

    double cpu_avertime=std::accumulate(cpu_times.begin()+1, cpu_times.end(), 0.0)/(cpu_times.size()-1);
    double gpu_avertime=std::accumulate(gpu_times.begin()+1, gpu_times.end(),0.0)/(gpu_times.size()-1);
    double mem_avertime=std::accumulate(mem_times.begin(), mem_times.end(),0.0)/(mem_times.size()-1);
    std::cout<<"cpu: "<<cpu_avertime<<" \tgpu: "<<gpu_avertime<<" \tmem_t: "<<mem_avertime<<std::endl;
}

