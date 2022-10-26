#include "cudaHeader.h"
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <cuda_runtime_api.h> 
#include <cuda.h> 
#include <cooperative_groups.h>


//CHECK FUNCTION
void checkStatus(cudaError_t cudaStatus, std::string err)
{
    if(cudaStatus != cudaSuccess)
    {
        std::cout << err << std::endl;
        exit(1);

    }
}


__global__ void cudaCalculateBestScore (char* cuda_seq1,char* proc_cuda_seq,int lenProcSeq,int** cuda_pointer_relation,int cuda_offset,int* maxScoreCuda)
{
    //blockIdx.x - index of block
    //blockDim.x - size of block
     __shared__ int maxScoreMutant; 
     __shared__ int maxScore;  
    maxScore=INT_MIN;

    int currentOffset=blockIdx.x;
    int currentMutant=threadIdx.x+1; //+1 because the tread id start from 0 
	int currnetScoreCuda=0;

    //CALCULATE SCORE
    for (int i = 0; i < lenProcSeq; i++)
    {
            if (i < currentMutant){
                currnetScoreCuda += cuda_pointer_relation[proc_cuda_seq[i]- 'A'][cuda_seq1[i + currentOffset]- 'A'];
            }
            else{
                currnetScoreCuda += cuda_pointer_relation[proc_cuda_seq[i]- 'A'][cuda_seq1[i + currentOffset + 1]- 'A'];
            }
    }
 

    __syncthreads();
    //CHECK MAX VALUE INSIDE EACH BLOCK - ATOMIC FOR PREVENT DATA RACE 
    atomicMax(&maxScore,currnetScoreCuda);
	__syncthreads(); //TO MAKE SURE THE MAX 
		

    //UPDATE MAX MUTANT INSIDE EACH BLOCK OF THE SAME MAX VALUE FROM LINE 49 - ATOMIC FOR PREVENT DATA RACE    
    if(currnetScoreCuda==maxScore){
        atomicExch(&maxScoreMutant,currentMutant); 
    }   
    __syncthreads();//TO MAKE SURE THE MAX MUTANT
	   
    //CHECK MAX VALUE BETWEEN BLOCKS AND COPY - ATOMIC FOR PREVENT DATA RACE
	if(currentMutant==1){
		atomicMax(maxScoreCuda,maxScore);
	}

    __syncthreads();
    //COPY RESULT
	if(currentMutant==1 && maxScore==maxScoreCuda[0] ){
        atomicExch(&maxScoreCuda[1],currentOffset);
        atomicExch(&maxScoreCuda[2],maxScoreMutant);
    }

    __syncthreads();

}  


void  cudaKernal(score_seq* alignments_score, char* seq1,char* proc_seq,int cuda_offset,int relation_group[][ALPHABET_SIZE]){

    //VARIABLES
    char* proc_cuda_seq;
    char* cuda_seq1;
    int* maxScoreCuda;
    cudaError_t cudaStatus;
    int numOfBlocks = cuda_offset;
    int lenSeq1=strlen(seq1);
    int lenProcSeq=strlen(proc_seq);
    size_t sizeSeq1 = (lenSeq1) * sizeof(char);
    size_t sizeProcSeq = (lenProcSeq) * sizeof(char);
    size_t sizeMaxScoreCuda = 3 * sizeof(int);



    cudaStatus = cudaMalloc((void**)&proc_cuda_seq,sizeProcSeq); //ALLOCATION cuda proc_seq
    checkStatus(cudaStatus,"Cuda Malloc Failed!");
    cudaStatus = cudaMemcpy(proc_cuda_seq,proc_seq,sizeProcSeq,cudaMemcpyHostToDevice); //MEMCOPY proc_seq
    checkStatus(cudaStatus, "Cuda MEMCPY failed!");
    
	cudaStatus = cudaMalloc((void**)&cuda_seq1,sizeSeq1); //ALLOCATION cuda seq1
    checkStatus(cudaStatus,"Cuda Malloc Failed!");
    cudaStatus = cudaMemcpy(cuda_seq1,seq1,sizeSeq1,cudaMemcpyHostToDevice); //MEMCOPY seq1
    checkStatus(cudaStatus, "Cuda MEMCPY failed!");

	cudaStatus = cudaMalloc((void**)&maxScoreCuda,sizeMaxScoreCuda); //ALLOCATION
    checkStatus(cudaStatus,"Cuda Malloc Failed!");

    int temp=INT_MIN;
    cudaStatus = cudaMemcpy(maxScoreCuda,&temp,sizeof(int),cudaMemcpyHostToDevice); //MEMCOPY FOR INITIALIZTION
    checkStatus(cudaStatus, "Cuda MEMCPY failed!");

    //ALLOCATION AND MEMCOPY FOR RELATION GROUP:
    int** cuda_relation_group=(int**)malloc(sizeof(int*)*ALPHABET_SIZE);
    for(int i=0; i<ALPHABET_SIZE;i++){
        cudaStatus=cudaMalloc(&cuda_relation_group[i],sizeof(int)*ALPHABET_SIZE);
        checkStatus(cudaStatus,"Cuda Malloc Failed!");
        cudaStatus = cudaMemcpy(cuda_relation_group[i],relation_group[i],sizeof(int)*ALPHABET_SIZE,cudaMemcpyHostToDevice); 
        checkStatus(cudaStatus,"Cuda Malloc Failed!");
    }
    
    int** cuda_pointer_relation;
    cudaStatus=cudaMalloc(&cuda_pointer_relation,ALPHABET_SIZE*sizeof(int*));
	checkStatus(cudaStatus,"Cuda Malloc Failed!");
	cudaStatus = cudaMemcpy(cuda_pointer_relation,cuda_relation_group,sizeof(int*)*ALPHABET_SIZE,cudaMemcpyHostToDevice);
    checkStatus(cudaStatus,"Cuda Malloc Failed!");

    //CALCULATE BEST SCORE CUDA:
    cudaCalculateBestScore<<<numOfBlocks,lenProcSeq>>>(cuda_seq1,proc_cuda_seq,lenProcSeq,cuda_pointer_relation,cuda_offset,maxScoreCuda);
    cudaStatus = cudaDeviceSynchronize();
    checkStatus(cudaStatus, "Cuda Failed!");

    //MEMCOPY RESULT:
    cudaStatus = cudaMemcpy(&alignments_score->mutant,&maxScoreCuda[2],sizeof(int),cudaMemcpyDeviceToHost); 
    cudaStatus = cudaMemcpy(&alignments_score->offset,&maxScoreCuda[1],sizeof(int),cudaMemcpyDeviceToHost); 
    cudaStatus = cudaMemcpy(&alignments_score->alignment_score,&maxScoreCuda[0],sizeof(int),cudaMemcpyDeviceToHost); 
    checkStatus(cudaStatus, "Cuda MEMCPY failed!");

    //FREE ALLOCATIONS:
    cudaStatus = cudaFree(proc_cuda_seq);
    checkStatus(cudaStatus,"Cuda Free Failed!");
    cudaStatus = cudaFree(cuda_seq1);
    checkStatus(cudaStatus,"Cuda Free Failed!");
	cudaStatus = cudaFree(maxScoreCuda);
    checkStatus(cudaStatus,"Cuda Free Failed!");


    for(int i=0; i<ALPHABET_SIZE;i++){
        cudaStatus=cudaFree(cuda_relation_group[i]);
        checkStatus(cudaStatus,"Cuda Free Failed!");
    }

	cudaStatus = cudaFree(cuda_pointer_relation);
    checkStatus(cudaStatus,"Cuda Free Failed!");
	free(cuda_relation_group);

}


