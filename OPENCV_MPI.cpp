#include <iostream>
#include <time.h>

#include "mpi.h"
#include "opencv2/opencv.hpp"

using namespace std;
using namespace cv;

int main(int argc, char** argv)
{
    Mat image;
    Mat outImage;
    size_t imageTotalSize;// calkowity rozmiar macierzy (rows * columns * channels)
    size_t imagePartialSize;
    int channels, k, imgCols;
    uchar* partialBuffer;
    clock_t start, end;
    
    //pomiar czasu START
    start = clock();

    // start MPI
    MPI_Init(&argc, &argv);


    int size, rank; 
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    Mat* partImage = new Mat[size];
    Mat* donePartImage = new Mat[size];

    //odczyt obrazu przez proces 0
    if (rank == 0)
    {   

        image = imread("test.jpg", IMREAD_UNCHANGED);
        //imshow("image IN", image);

        if (image.empty())
        {
            cerr << "Brak obrazu!!!" << endl;
            return -1;
        }
        channels = image.channels();
        imageTotalSize = image.step[0] * image.rows;
        imgCols = image.cols;
    

        //sprawdzanie mozliwosci rownomiernego rozlozenia na procesory
        if (image.total() % size)
        {
            cerr << "Nie można równomiernie podzielić obrazu między procesy. Wybierz inną liczbę procesów!" << endl;
            return -2;
        }

        //liczba bajtow wysylana do kazdego procesu
        imagePartialSize = imageTotalSize / size;
        cout << "Obraz zostanie podzielony na bloki po " << imagePartialSize << " B" << endl;
    }

    //wysylanie informacji o rozmiarze czesci obrazu do wszystkich procesow w grupie
    MPI_Bcast(&imagePartialSize, 1, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);
    MPI_Bcast(&channels, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&imgCols, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);//synchronizacja
    
    //czesci obrazu w procesach
    partialBuffer = new uchar[imagePartialSize];
    MPI_Barrier(MPI_COMM_WORLD);//synchronizacja

    // wysylanie czesci obrazu do wszystkich procesow w grupie
    MPI_Scatter(image.data, imagePartialSize, MPI_UNSIGNED_CHAR, partialBuffer, imagePartialSize, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);//synchronizacja
    
    //przetwarzanie obrazu
    int Filter[49] = { 1,1,2,2,2,1,1,1,2,2,4,2,2,1,2,2,4,8,4,2,2,2,4,8,16,8,4,2,2,2,4,8,4,2,2,1,2,2,4,2,2,1,1,1,2,2,2,1,1 }; //wygladzanie Gaussowskie
    int nom = 0;
    for (int i = 0; i < 49; i++) nom += Filter[i];

    uchar* B, * G, * R, * Bpx, * Gpx, * Rpx;
    int step, px = imgCols * 3, Bsum, Gsum, Rsum;

    for (int i = 0; i < imagePartialSize; i += channels)
    {
        B = &partialBuffer[i];
        G = &partialBuffer[i + 1];
        R = &partialBuffer[i + 2];
        Bsum = 0; 
        Gsum = 0; 
        Rsum = 0;

        step = (-9);
        for (int j = 0; j < 49; j++ )
        {

            if (((i - 3 * px + step) >= 0) && ((i + 2 + 3 * px + step) <= (int)imagePartialSize))
            {

                if (j < 7)
                {
                    
                    Bpx = &partialBuffer[i - 3 * px + step];
                    Gpx = &partialBuffer[i + 1 - 3 * px + step];
                    Rpx = &partialBuffer[i + 2 - 3 * px + step];

                    Bsum += Filter[j] * (int)(*Bpx);
                    Gsum += Filter[j] * (int)(*Gpx);
                    Rsum += Filter[j] * (int)(*Rpx);
                }
                else if ((j < 14) && (j >= 7))
                {

                    Bpx = &partialBuffer[i - 2 * px + step];
                    Gpx = &partialBuffer[i + 1 - 2 * px + step];
                    Rpx = &partialBuffer[i + 2 - 2 * px + step];

                    Bsum += Filter[j] * (int)(*Bpx);
                    Gsum += Filter[j] * (int)(*Gpx);
                    Rsum += Filter[j] * (int)(*Rpx);
                }
                else if ((j < 21) && (j >= 14))
                {

                    Bpx = &partialBuffer[i - px + step];
                    Gpx = &partialBuffer[i + 1 - px + step];
                    Rpx = &partialBuffer[i + 2 - px + step];

                    Bsum += Filter[j] * (int)(*Bpx);
                    Gsum += Filter[j] * (int)(*Gpx);
                    Rsum += Filter[j] * (int)(*Rpx);
                }
                else if ((j < 28) && (j >= 21))
                {
                    
                    Bpx = &partialBuffer[i + step];
                    Gpx = &partialBuffer[i + 1 + step];
                    Rpx = &partialBuffer[i + 2 + step];

                    Bsum += Filter[j] * (int)(*Bpx);
                    Gsum += Filter[j] * (int)(*Gpx);
                    Rsum += Filter[j] * (int)(*Rpx);
                }
                else if ((j < 35) && (j >= 28))
                {

                    Bpx = &partialBuffer[i + px + step];
                    Gpx = &partialBuffer[i + 1 + px + step];
                    Rpx = &partialBuffer[i + 2 + px + step];

                    Bsum += Filter[j] * (int)(*Bpx);
                    Gsum += Filter[j] * (int)(*Gpx);
                    Rsum += Filter[j] * (int)(*Rpx);
                }
                else if ((j < 42) && (j >= 35))
                {

                    Bpx = &partialBuffer[i + 2 * px + step];
                    Gpx = &partialBuffer[i + 1 + 2 * px + step];
                    Rpx = &partialBuffer[i + 2 + 2 * px + step];

                    Bsum += Filter[j] * (int)(*Bpx);
                    Gsum += Filter[j] * (int)(*Gpx);
                    Rsum += Filter[j] * (int)(*Rpx);
                }
                else if (j >= 42)
                {
                    
                    Bpx = &partialBuffer[i + 3 * px + step];
                    Gpx = &partialBuffer[i + 1 + 3 * px + step];
                    Rpx = &partialBuffer[i + 2 +  3 * px + step];

                    Bsum += Filter[j] * (int)(*Bpx);;
                    Gsum += Filter[j] * (int)(*Gpx);
                    Rsum += Filter[j] * (int)(*Rpx);
                }

                *B = (Bsum / nom);
                *G = (Gsum / nom);
                *R = (Rsum / nom);

                if (((int)*B) > 255) *B = 255;
                else if (((int)*B) < 0) *B = 0;
                if (((int)*G) > 255) *G = 255;
                else if (((int)*G) < 0) *G = 0;
                if (((int)*R) > 255) *R = 255;
                else if (((int)*R) < 0) *R = 0;

                step += 3;
                if (step > 9) step = (-9);
                
            }  
        }
    }
    for (int i = 0; i < imagePartialSize; i += channels)
    {
        B = &partialBuffer[i];
        G = &partialBuffer[i + 1];
        R = &partialBuffer[i + 2];

        if ((i - 3*px) < 0)
        {
            *B = (int)partialBuffer[i + 3 * px];
            *G = (int)partialBuffer[i + 1 + 3 * px];
            *R = (int)partialBuffer[i + 2 + 3 * px];
        }
        else if ((i + 3*px) > (int)imagePartialSize)
        {
            *B = (int)partialBuffer[i - 3 * px];;
            *G = (int)partialBuffer[i + 1 - 3 * px];
            *R = (int)partialBuffer[i + 2 - 3 * px];
        }
    }
    
    MPI_Barrier(MPI_COMM_WORLD);//synchronizacja

    //tworzenie zmiennej przechowujaca nowy obraz
    if (rank == 0)
    {
        outImage = Mat(image.size(), image.type());
    }

    //zbieranie informacji od wszystkich do procesu 0
    MPI_Gather(partialBuffer, imagePartialSize, MPI_UNSIGNED_CHAR, outImage.data, imagePartialSize, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);


    //zapisywanie i wyswietlanie nowego obrazu
    if (rank == 0)
    {
        imwrite("new_image.jpg", outImage);
        
        //Pomiar czasu STOP
        end = clock();
        double elapsed = double(end - start) / CLOCKS_PER_SEC;
        printf("\nZmierzony czas: %.3f s.\n", elapsed);

        /*while (true)
        {
            imshow("image OUT", outImage);

            if (waitKey(1) == 27)
                break;
        }*/
        destroyAllWindows();

    }
    delete[]partialBuffer;

    MPI_Finalize();
    
    

    return 0;
}