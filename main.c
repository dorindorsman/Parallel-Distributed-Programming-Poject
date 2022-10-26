#include <mpi.h>
#include <omp.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>
#include <string.h>
#include <time.h>
#include <ctype.h>
#include "cudaHeader.h"


//CONSTANTS
#define ROOT 0
#define GROUP_ONE_SIZE 9
#define GROUP_TWO_SIZE 11
#define ALPHABET_SIZE 26
#define MAX_SEQ1_SIZE 3000
#define MAX_SEQ_SIZE  2000
#define WEIGHTS 4

enum tags{WORK,STOP};


//ARRAYS
int weights[WEIGHTS];

const char* group_one[] = { "NDEQ", "NEQK", "STA", "MILV",
 "QHRK", "NHQK","FYW", "HY", "MILF" };

 const char* group_two[] = { "SAG", "ATV", "CSA", "SGND",
 "STPA", "STNK","NEQHRK", "NDEQHK", "SNDEQK", "HFY", "FVLIM" };

int relation_group[ALPHABET_SIZE][ALPHABET_SIZE]={0};


//ANNOUNCEMENT OF FUNCTIONS:
void bulidRelationGroup();
int inGroup(char c1, char c2, const char **group, int size);
int calc_num_offsets(char *seq1, char *seq2);
void masterProcess(score_seq *alignments_scores, char *seq1, char **seqs, int number_of_sequences, int num_procs);
void workerProcess(char *seq1, int number_of_sequences, int num_procs);


//FUNCTIONS:
void bulidRelationGroup()
{
    for (char i = 'A'; i <= 'Z'; i++)
    {
        for (char j = 'A'; j <= 'Z'; j++)
        {
            if (i == j)
            {
                relation_group[i - 'A'][j - 'A'] = weights[0];
            }
            else if (inGroup(i, j, group_one, GROUP_ONE_SIZE))
            {
                relation_group[i - 'A'][j - 'A'] = -1 * weights[1];
            }
            else if (inGroup(i, j, group_two, GROUP_TWO_SIZE))
            {
                relation_group[i - 'A'][j - 'A'] = -1 * weights[2];
            }
            else
            {
                relation_group[i - 'A'][j - 'A'] = -1 * weights[3];
            }
        }
    }

}

int inGroup(char c1, char c2, const char **group, int size)
{
    for (int t = 0; t < size; t++)
    {
        if (strchr(group[t], c1) != NULL && strchr(group[t], c2) != NULL)
            return 1;
    }
    return 0;
}

// Returns the subtraction of the length between the two seq = num of offsets
int calc_num_offsets(char *seq1, char *seq2)
{
    int len1 = (int)(size_t)strlen(seq1);
    int len2 = (int)(size_t)strlen(seq2);
    return len1 - len2;
}

//MASTER PROCESS:
void masterProcess(score_seq *alignments_scores, char *seq1, char **seqs, int number_of_sequences, int num_procs)
{

    MPI_Status status;
    int process_id;
    int jobs_sent = 0, job_rec = 0, num_workers = num_procs - 1, job_id = 0, tag = WORK;

    //START WORK FOR PROCESS-SEND ONE WORK TO EACH PROCESS
    for (process_id = 1; process_id < num_procs; process_id++)
    {
        MPI_Send(seqs[jobs_sent], MAX_SEQ_SIZE, MPI_CHAR, process_id, WORK, MPI_COMM_WORLD);
        MPI_Send(&jobs_sent, 1, MPI_INT, process_id, WORK, MPI_COMM_WORLD);
        jobs_sent++;
    }

    //CASE : WHEN WE HAVE MORE SEQ THEN PROCESS
    while (jobs_sent < number_of_sequences)
    {
        if (jobs_sent >= number_of_sequences - num_workers)
        {
            tag = STOP;
        }

        //GET THE DATA AND SEND MORE
        MPI_Recv(&job_id, 1, MPI_INT, MPI_ANY_SOURCE, WORK, MPI_COMM_WORLD, &status);
        MPI_Recv(&alignments_scores[job_id].alignment_score, 1, MPI_INT, MPI_ANY_SOURCE, WORK, MPI_COMM_WORLD, &status);
        MPI_Recv(&alignments_scores[job_id].mutant, 1, MPI_INT, MPI_ANY_SOURCE, WORK, MPI_COMM_WORLD, &status);
        MPI_Recv(&alignments_scores[job_id].offset, 1, MPI_INT, MPI_ANY_SOURCE, WORK, MPI_COMM_WORLD, &status);
        job_rec++;
        MPI_Send(seqs[jobs_sent], MAX_SEQ_SIZE, MPI_CHAR, status.MPI_SOURCE, tag, MPI_COMM_WORLD);
        MPI_Send(&jobs_sent, 1, MPI_INT, status.MPI_SOURCE, tag, MPI_COMM_WORLD);
        jobs_sent++;
    }

    //THE LAST WORK
    while (job_rec < number_of_sequences)
    {
        MPI_Recv(&job_id, 1, MPI_INT, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
        MPI_Recv(&alignments_scores[job_id].alignment_score, 1, MPI_INT, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
        MPI_Recv(&alignments_scores[job_id].mutant, 1, MPI_INT, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
        MPI_Recv(&alignments_scores[job_id].offset, 1, MPI_INT, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
        job_rec++;
    }
}

//WORKER PROCESS:
void workerProcess(char *seq1, int number_of_sequences, int num_procs)
{

    //VARIABLES
    MPI_Status status;
    int tag = 0;
    int num_Of_Threads = omp_get_max_threads(); //HOW MUCH THREADS
    int tid;
    int job_rec = 0;
    int alignment_score = 0;
    int num_of_offset = 0;
    char proc_seq[MAX_SEQ_SIZE];
    int len_seq=0;
    score_seq final_score; //THE BEST SCORE FOR EACH SEQ
    score_seq CUDA_score;  //THE BEST SCORE FOR EACH SEQ IN CUDA
    score_seq *OMP_scores = (score_seq *)malloc(sizeof(score_seq) * num_Of_Threads); //THE BEST SCORE FOR EACH SEQ IN OMP
    int scoreOMP;  //TEMP VARIABLE  SCORE FOR CHECK IN OMP

    do
    {
            //INITIALIZTION:
            for (int i = 0; i < num_Of_Threads; i++)
            {
                OMP_scores[i].alignment_score = INT_MIN;
            }

            final_score.alignment_score=INT_MIN;
            CUDA_score.alignment_score=INT_MIN;
            
        //GET THE WORK FROM MASTER
        MPI_Recv(&proc_seq, MAX_SEQ_SIZE, MPI_CHAR, ROOT, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
        len_seq=strlen(proc_seq);
        MPI_Recv(&job_rec, 1, MPI_INT, ROOT, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
        tag = status.MPI_TAG;

        //CALCULATE WORK BETWEEN OMP AND CUDA
        num_of_offset = calc_num_offsets(seq1, proc_seq);
        int cuda_offset = num_of_offset / 2;
        int omp_offset = num_of_offset / 2 + (num_of_offset % 2);
        int n=cuda_offset;
        

        //There is a maximum of 1024 threads per block:
        if(len_seq<1024){
            //CALCULATING THE OFFSETS OF THE FIRST HALF WITH CUDA:
            cudaKernal(&CUDA_score, seq1, proc_seq, cuda_offset,relation_group);
        }else{
            //CALCULATING ALL THE OFFSET WITH OMP:
            n=0;
        }

        //CALCULATING THE OFFSETS OF THE SECOND HALF WITH OMP: if n=cuda_offset 
        //CALCULATING ALL THE OFFSET WITH OMP: if n=0

//n=j -> INDEX OF OFFSET
#pragma omp parallel for private(scoreOMP,tid) shared(seq1,num_of_offset,OMP_scores)
        for (int j=n; j <num_of_offset; j++)
        {
            tid = omp_get_thread_num(); //GET THREAD ID
            
            //k -> INDEX OF '-' Mutant
            for (int k = 1; k <= len_seq; k++)
            {
                scoreOMP = 0;
                for (int i = 0; i < len_seq; i++)
                {
                    if (i < k)
                    {
                        scoreOMP += relation_group[proc_seq[i]- 'A'][seq1[i + j]- 'A'];
                    }
                    else
                    {
                        scoreOMP += relation_group[proc_seq[i]- 'A'][seq1[i + j + 1]- 'A'];
                    }

                }
                //CHECK MAX FOR EACH THRED   
                if (scoreOMP > OMP_scores[tid].alignment_score)
                {
                    OMP_scores[tid].alignment_score = scoreOMP;
                    OMP_scores[tid].mutant = k;
                    OMP_scores[tid].offset = j;
                }
            }
        }

        //CHECK MAX BETWEEN RESULTS OF EACH THRED
        int max = INT_MIN;
        for (int i = 0; i < num_Of_Threads; i++)
        {
            if (max < OMP_scores[i].alignment_score)
            {
                final_score.alignment_score = OMP_scores[i].alignment_score;
                final_score.mutant = OMP_scores[i].mutant;
                final_score.offset = OMP_scores[i].offset;
                max = OMP_scores[i].alignment_score;
            }
        }
        //CHECK MAX BETWEEN OMP AND CUDA
        if (CUDA_score.alignment_score > final_score.alignment_score)
        {
            
            final_score.alignment_score = CUDA_score.alignment_score;
            final_score.mutant = CUDA_score.mutant;
            final_score.offset = CUDA_score.offset;
        }

        //SEND RESULTS TO MASTER
        MPI_Send(&job_rec, 1, MPI_INT, status.MPI_SOURCE, tag, MPI_COMM_WORLD);
        MPI_Send(&final_score.alignment_score, 1, MPI_INT, status.MPI_SOURCE, tag, MPI_COMM_WORLD);
        MPI_Send(&final_score.mutant, 1, MPI_INT, status.MPI_SOURCE, tag, MPI_COMM_WORLD);
        MPI_Send(&final_score.offset, 1, MPI_INT, status.MPI_SOURCE, tag, MPI_COMM_WORLD);

    } while (tag != STOP);
    //FREE ALLOCATIONS
    free(OMP_scores);
}

int main(int argc, char *argv[])
{
    //GENERAL VARIABLES
    double start = time(NULL);
    int tag;
    int my_rank;
    int num_procs;
    int number_of_sequences;
    char *seq1 = (char *)malloc(sizeof(char) * MAX_SEQ1_SIZE);
    char **seqs;



    //MPI INIT
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    //PROCESS 0
    //SCAN THE DATA - READS VARIABLES FROM input.txt
    if (my_rank == ROOT)
    {
        FILE *file_input;
        file_input = fopen("input.txt", "r");
        if (!file_input)
        {
            printf("File open failed");
        }

        //SCAN WEIGHTS
        scanf("%d %d %d %d", &weights[0], &weights[1], &weights[2], &weights[3]);

        //SCAN SEQ1
        scanf("%s", seq1);

        //SCAN NUM OF SEQS
        scanf("%d", &number_of_sequences);

        //ALLOCATE
        seqs = (char **)malloc(sizeof(char *) * number_of_sequences);
        if (!seqs)
        {
            printf("malloc Failed! (line 254)\n");
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }

        //SCAN SEQS
        for (int i = 0; i < number_of_sequences; i++)
        {
            seqs[i] = (char *)malloc(sizeof(char) * MAX_SEQ_SIZE);
            if (!seqs[i])
            {
                printf("malloc Failed! (line 56 i=%d)\n", i);
                MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
            }
            scanf("%s", seqs[i]);
        }

        //CHECK FOR EACH PROCESS HAVE DATA TO DO
        if (number_of_sequences < num_procs - 1)
        {
            printf("The program needs at least the %d inputs to work\n", num_procs - 1);
            fclose(file_input); //CLOSE FILE
            MPI_Finalize();
            return 0;
        }

        fclose(file_input); //CLOSE FILE
    }

    
    //GENERAL VARIABLES
    score_seq *alignments_scores = (score_seq *)malloc(sizeof(score_seq) * number_of_sequences);
    for (int i = 0; i < number_of_sequences; i++)
    {
        alignments_scores[i].alignment_score = 0;
        alignments_scores[i].offset = 0;
        alignments_scores[i].mutant = 0;
    }


   //BUILD RELATIONS
    bulidRelationGroup();

    //BROADCAST TO ALL PROC:
    MPI_Bcast(weights, WEIGHTS, MPI_INT, ROOT, MPI_COMM_WORLD);        // MAKE SURE EVERY PROC KNOWS THE WEIGHTS
    MPI_Bcast(seq1, MAX_SEQ1_SIZE, MPI_CHAR, ROOT, MPI_COMM_WORLD);    // MAKE SURE EVERY PROC KNOWS SEQ1
    MPI_Bcast(&number_of_sequences, 1, MPI_INT, ROOT, MPI_COMM_WORLD); // MAKE SURE EVERY PROC KNOWS NUMBER OF SEQ
    MPI_Bcast(relation_group,ALPHABET_SIZE*ALPHABET_SIZE,MPI_INT,ROOT,MPI_COMM_WORLD); // MAKE SURE EVERY PROC KNOWS relation_group



    //PROCESS 0
    if (my_rank == ROOT)
    {
        masterProcess(alignments_scores, seq1, seqs, number_of_sequences, num_procs);
    }

    //WORKERS PROCESS
    if (my_rank != ROOT)
    {
        workerProcess(seq1, number_of_sequences, num_procs);
    }



    if (my_rank == ROOT)
    {
    //PRINT RESULTS:

        //printf("SEQ1: %s \n", seq1);
        printf("SEQ1:  \n");
        printf("BEST SCORES FOR EACH SEQ:\n");
    
        for (int i = 0; i < number_of_sequences; i++)
        {
            printf("SEQ%d:\n",i+2);
            printf("OFFSET (n) = %d , MUTANT (k) = %d , SCORE = %d \n", alignments_scores[i].offset , alignments_scores[i].mutant, alignments_scores[i].alignment_score);
            printf("\n");
        }    
    
    double lentime = time(NULL)-start;
    printf("time: %lf \n",lentime);


    //FREE ALLOCATION:
        for (int i = 0; i < number_of_sequences; i++)
        {
            free(seqs[i]);
        }
        free(seqs);
    }

    free(alignments_scores);
    free(seq1);
    MPI_Finalize();
    return 0;
}
