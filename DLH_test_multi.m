clc;clear all;
mkdir test;
filename_test_img='.\MNIST_data\t10k-images.idx3-ubyte';
filename_test_lab='.\MNIST_data\t10k-labels.idx1-ubyte';
test_img = loadMNISTImages(filename_test_img); 
test_lab = loadMNISTLabels(filename_test_lab);
num_sum=64;multi_x=2;multi_y=2;multi=multi_x*multi_y;
num_test=1;%num_sub=1000;
sum_test=num_test*multi;
d=30;length_z=20;
n = 4;                                  %需要分成n个色阶
STEP = 256/n;                            %每个色阶宽度
Lamda=632.8e-6;                          %波长
dL=10.8e-3;                              %DMD像素间距
k=2*pi/Lamda;
test_amp_multi=zeros(num_sum,num_sum,multi);
test_amp_posit=zeros(num_sum,num_sum,multi);
test_input=zeros(num_test,num_sum,num_sum);
HoloAmp=zeros(num_test,num_sum,num_sum);
N=num_sum;M=num_sum;
[fx,fy]=meshgrid(linspace(-1/(2*dL),1/(2*dL),M),linspace(-1/(2*dL),1/(2*dL),N));
%设置single-sideband fil ter
single_filter=ones(N,M);
for ii=1:N/2
    single_filter(ii,:)=0;
end
N_sub=28;M_sub=28;
for num_img=1:multi:sum_test
    test_input_amp=zeros(num_sum,num_sum);
    test_input_depth=zeros(num_sum,num_sum);
    for num_sub_x=1: multi_x
        for num_sub_y=1:multi_y
            test_amp_sub=test_img(:,:,num_img+num_sub_x*num_sub_y-1);
            test_amp_sub_binary=test_amp_sub;
            test_amp_sub_binary(test_amp_sub_binary>0)=1;
            tran_amp_depth_sub=zeros(N_sub,M_sub)+unidrnd(4)*STEP*test_amp_sub_binary;
            %tran_amp_depth_sub=zeros(N_sub,M_sub)+1*STEP*test_amp_sub_binary;
            test_amp_sub_posit_x=(num_sum-multi_x*M_sub)/multi_x/2+num_sub_x*M_sub/2+(num_sub_x-1)...
                                   *((num_sum-multi_x*M_sub)/multi_x+M_sub/2);
            test_amp_sub_posit_y=(num_sum-multi_y*N_sub)/multi_y/2+num_sub_y*N_sub/2+(num_sub_y-1)...
                                   *((num_sum-multi_y*N_sub)/multi_y+N_sub/2);
            test_input_amp(test_amp_sub_posit_y-N_sub/2:test_amp_sub_posit_y+N_sub/2-1,...
                            test_amp_sub_posit_x-M_sub/2:test_amp_sub_posit_x+M_sub/2-1)...
                            =test_amp_sub;
            test_input_depth(test_amp_sub_posit_y-N_sub/2:test_amp_sub_posit_y+N_sub/2-1,...
                            test_amp_sub_posit_x-M_sub/2:test_amp_sub_posit_x+M_sub/2-1)...
                            =tran_amp_depth_sub;
        end
    end
num_img_list=(num_img-1)/multi+1;
%test_input_depth=zeros(num_sum,num_sum)+1;
% test_input_img=(d/length_z*n*256+test_input_depth/STEP*256+test_input_amp)/...
%                                            ((d+length_z)/length_z*n*256);
test_input_img=(test_input_depth/STEP*256+test_input_amp)/...
                                            ((n+1)*256);
%test_input_img=(d/length_z*n*256+test_input_depth*256+test_input_amp);
%test_input_img_gray=mat2gray(test_input_img);
%test_input_amp_gray=mat2gray(test_input_amp);
test_input(num_img_list,:,:)=test_input_img;
%test_input(num_img,:,:)=test_input_img_gray;
%test_input(num_img,:,:)=test_input_amp_gray;

AnDiffract=0;
POSITION=test_input_amp;
DEPTH=test_input_depth;
OUTPUT = zeros(N,M,n);
 for I = 1:n
     [x,y]=find(DEPTH>(I-1)*STEP & DEPTH<=I*STEP);
     for J=1:length(x)
         OUTPUT(x(J),y(J),I)=POSITION(x(J),y(J));
     end     
    A=OUTPUT(:,:,I);
    %A=im2double(A);
    oz=d+length_z*I/n;
    %A1=A.*exp(1i*2*pi*rand(N,M));
    A1=A;
    HFunct=exp(1i*k*oz.*sqrt(1-(Lamda*fx).^2-(Lamda*fy).^2)); 
    DScreen_F=fftshift(fft2(fftshift(A1))); 
    %DScreen_F=DScreen_F.*single_filter;
    AnDiffract=fftshift(ifft2(fftshift(DScreen_F.*HFunct)))+AnDiffract;
 end


AnF=fftshift(fft2(fftshift(AnDiffract))); 
%figure,imshow(abs(AnF),[]);
AnF=AnF.*single_filter;
%figure,imshow(abs(AnF),[]);
AnDiffract=fftshift(ifft2(fftshift(AnF)));
HoloAmp_img=2*real(AnDiffract);
HoloAmp_img_gray=mat2gray(HoloAmp_img);
HoloAmp(num_img_list,:,:)=HoloAmp_img_gray;
nameHoloAmp_img_gray=strcat(['G:\Research\Deep learning holography\Deep-Speckle-Correlation-master'...
                       '\test\momd\'],'HoloAmp_',num2str(num_img_list),'.bmp');
figure(1),imshow(HoloAmp_img_gray);
imwrite(HoloAmp_img_gray,nameHoloAmp_img_gray); 
test_input_depth_gray=mat2gray(test_input_depth);
nametest_input_depth_gray=strcat(['G:\Research\Deep learning holography\Deep-Speckle-Correlation-master'...
                       '\test\momd\'],'test_input_depth_',num2str(num_img_list),'.bmp');
figure(3),imshow(test_input_depth_gray); 
imwrite(test_input_depth_gray,nametest_input_depth_gray);
test_input_amp_gray=mat2gray(test_input_amp);
nametest_input_amp_gray=strcat(['G:\Research\Deep learning holography\Deep-Speckle-Correlation-master'...
                       '\test\momd\'],'test_input_amp_',num2str(num_img_list),'.bmp');
figure(2),imshow(test_input_amp_gray);
imwrite(test_input_amp_gray,nametest_input_amp_gray); 
end
path='G:\Research\Deep learning holography\Deep-Speckle-Correlation-master\test\momd\';
save ([path 'test_input_true'],'test_input','-v7.3');
path='G:\Research\Deep learning holography\Deep-Speckle-Correlation-master\test\momd\';
save ([path 'test_output_true'],'HoloAmp','-v7.3');
% An=mat2gray(HoloAmp_img);
% nameAn=strcat('.\Image\','holo_','.bmp');
% imwrite(An,nameAn); 
% imshow(An);