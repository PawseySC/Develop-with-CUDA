.PHONY = default clean

default: transfer.cu
	nvcc -o transfer transfer.cu

clean : transfer
	rm -f transfer transfer.txt slurm*.out make-transfer.pbs.* run-transfer.pbs.*
