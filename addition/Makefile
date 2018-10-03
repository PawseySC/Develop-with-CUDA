.PHONY = default clean

default: addition.cu
	nvcc -o addition addition.cu

clean :
	rm -f addition addition.txt slurm*.out make-addition.pbs.* run-addition.pbs.*
