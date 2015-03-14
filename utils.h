#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

/*This is part of the implementation of the paper "Text Detection and Character Recognition with
  Unsupervised Feature Learning" by A. Coates et al. in ICDAR2011*/

// normalize for contrast and apply ZCA whitening to a set of image patches
void normalizeAndZCA(Mat& patches, Mat& M, Mat&P)
{

  //Normalize for contrast
  for (int i=0; i<patches.rows; i++)
  {
    Scalar row_mean, row_std;
    meanStdDev(patches.row(i),row_mean,row_std);
    row_std[0] = sqrt(pow(row_std[0],2)*patches.cols/(patches.cols-1)+10);
    patches.row(i) = (patches.row(i) - row_mean[0]) / row_std[0];
  }

  
  //ZCA whitening
  if ((M.dims == 0) || (P.dims == 0))
  {
    Mat CC;
    calcCovarMatrix(patches,CC,M,COVAR_NORMAL|COVAR_ROWS|COVAR_SCALE);
    CC = CC * patches.rows / (patches.rows-1);
   
  
    Mat e_val,e_vec;
    eigen(CC.t(),e_val,e_vec);
    e_vec = e_vec.t();
    sqrt(1./(e_val + 0.1), e_val);
  
  
    Mat V = Mat::zeros(e_vec.rows, e_vec.cols, CV_64FC1);
    Mat D = Mat::eye(e_vec.rows, e_vec.cols, CV_64FC1);
  
    for (int i=0; i<e_vec.cols; i++)
    {
      e_vec.col(e_vec.cols-i-1).copyTo(V.col(i));
      D.col(i) = D.col(i) * e_val.at<double>(0,e_val.rows-i-1);
    }
  
    P = V * D * V.t();
  }

  for (int i=0; i<patches.rows; i++)
    patches.row(i) = patches.row(i) - M;

  patches = patches * P;

}

//Uses dot-product Kmeans to learn a specified number of bases
void run_projection_kmeans(Mat& patches, Mat& centroids, int K, int n_iter)
{

  //randomly initialize centroids
  centroids = Mat(K, patches.cols, CV_64FC1);
  randn(centroids, Mat::zeros(1,1,CV_64FC1), Mat::ones(1,1,CV_64FC1));
  //normalize all centroids
  Mat rowSum = Mat::zeros(centroids.rows,1, CV_64FC1);
  reduce(centroids.mul(centroids), rowSum, 1, REDUCE_SUM);
  cv::sqrt(rowSum,rowSum);
  for (int r=0; r<centroids.rows; r++)
    centroids.row(r) = centroids.row(r) / rowSum.at<double>(r,0);

  int batch_size=1000;
  for (int itr=0; itr<n_iter; itr++)
  {
    cout << "K-means iteration " << itr << "/" << n_iter << endl;
    Mat summation = Mat::zeros(K, patches.cols, CV_64FC1);
    Mat counts = Mat::zeros(K,1,CV_64FC1);
    for (int i=0; i<patches.rows; i=i+batch_size)
    {
      int lastIndex = min(i+batch_size-1, patches.rows);
      int m = lastIndex - i + 1;
      Mat tmp;
      patches(Rect(0,i,patches.cols,lastIndex-i)).copyTo(tmp);
      Mat projection =  centroids * tmp.t();
      tmp = Mat::zeros(projection.cols, projection.rows, CV_64FC1);
      for (int c=0; c<projection.cols; c++)
      {
        double minVal, maxVal;
        Point minLoc, maxLoc;
        minMaxLoc(abs(projection.col(c)),&minVal,&maxVal,&minLoc,&maxLoc);
        tmp.at<double>(c,(int)maxLoc.y) = 1.0;
      }
      Mat S = projection.mul(tmp.t());
      patches(Rect(0,i,patches.cols,lastIndex-i)).copyTo(tmp);
      summation = summation + S*tmp;
      rowSum = Mat::zeros(S.rows,1, CV_64FC1);
      reduce(S, rowSum, 1, REDUCE_SUM);
      counts = counts + rowSum;
    }

    //normalize all centroids
    rowSum = Mat::zeros(summation.rows,1, CV_64FC1);
    reduce(summation.mul(summation), rowSum, 1, REDUCE_SUM);
    cv::sqrt(rowSum,rowSum);
    for (int r=0; r<summation.rows; r++)
    {
      if (counts.at<double>(r,0) == 0)
      {
        //just to zap empty D so they don't introduce NaNs everywhere.
        centroids.row(r) = Scalar(0);
      }
      else
        centroids.row(r) = summation.row(r) / rowSum.at<double>(r,0);
    }
  }


  //This uses the normal K-means algorithm in OpenCV
  /*cout << "Start clustering " << patches.rows << " patches ... " << endl;
  patches.convertTo(patches, CV_32FC1);
  Mat centroids,labels;
  kmeans(patches, K, labels, TermCriteria( TermCriteria::EPS+TermCriteria::COUNT, 100, 0.01), 
         3, KMEANS_PP_CENTERS, centroids);*/
  
  cout << "Done!" << endl;

}

// remove centroids whos variance is lower than varthresh
void selectCentroids(Mat& centroids, double varthresh)
{

  int patch_size = centroids.cols;

  // remove centroids whos variance is lower than varthresh
  vector<double> new_centroids;

  for (int i = 0; i<centroids.rows; i++)
  {
      Mat tmp;
      centroids.row(i).copyTo(tmp);
      double minVal, maxVal;
      minMaxLoc(tmp,&minVal,&maxVal);
      tmp = (tmp - minVal) / (maxVal - minVal);

      Scalar row_mean, row_std;
      meanStdDev(tmp,row_mean,row_std);
      row_std[0] = pow(row_std[0],2)*tmp.cols/(tmp.cols-1);
      if (row_std[0]>varthresh)
      {
         vector<double> row;
         centroids.row(i).copyTo(row);
         new_centroids.insert(new_centroids.end(),row.begin(),row.end());
      }
  }

  centroids = Mat(new_centroids);
  centroids = centroids.reshape(1,(int)(new_centroids.size()/patch_size));
}

/*Visualize the filter bank*/
void visualizeNatwork(Mat& centroids)
{

  int K = centroids.rows;
  int patch_width, patch_height;
  patch_width = patch_height = sqrt(centroids.cols);
  Mat filters = Mat::zeros(patch_height*((K/16)+1)+((K/16)+1)+1,patch_width*16+17,CV_8UC1);

  for(int i=0; i<centroids.rows; i++)
  {
    Mat filter;
    centroids.row(i).copyTo(filter);
    double minVal, maxVal;
    minMaxLoc(filter,&minVal,&maxVal);
    filter = ((filter - minVal) / (maxVal-minVal)) *255;
    filter.convertTo(filter, CV_8UC1);
    filter = filter.reshape(0,8);
    int x = (i%16) * 8 + (i%16+1);
    int y = (i/16) * 8 + (i/16+1);
    filter.copyTo(filters(Rect(x,y,8,8)));
  }

  Mat filters_large = Mat::zeros(filters.rows*5, filters.cols*5, CV_8UC1);

  for (int x=0; x<filters.cols; x++)
  {
    for (int y=0; y<filters.rows; y++)
    {
      int value = filters.at<unsigned char>(y,x);
      filters_large(Rect(x*5,y*5,5,5)) = value;
    }
  }

  imwrite("filters.jpg",filters_large);
  imshow("filters",filters_large);
  waitKey(0);
}

