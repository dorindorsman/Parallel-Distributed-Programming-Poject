
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <limits.h>
#include <iostream>

#define ALPHABET_SIZE 26
#define NUM_THREADS_PER_BLOCK 256

typedef struct{
	int mutant;
	int offset;
	int alignment_score;
}score_seq;



void cudaKernal(score_seq* alignments_score, char* seq1,char* proc_seq,int cuda_offset,int relation_group[][ALPHABET_SIZE]);


