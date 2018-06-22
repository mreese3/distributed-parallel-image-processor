#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <string.h>
#include <errno.h>
#include <mpi.h>
#include <opencv2/core/core_c.h>
#include <opencv2/highgui/highgui_c.h>

#define COMM_ROOT 0

void printerr(const char *s, ...)
{
	va_list args;
	va_start(args, s);
	vfprintf(stderr, s, args);
	fprintf(stderr,"\n");
	va_end(args);
	exit(EXIT_FAILURE);
}

void processImg(unsigned char *buf, size_t nbytes, size_t pixel_size)
{
	for (int i = 0; i < nbytes; i += pixel_size)
	{
		/*
		 * Explanation
		 * Observe the ratio between them (r / g), if (r/g) is between 0.75 and 1.5, the color is yellow, provided blue is low
		 */
		float rgratio = (float)buf[i+2] / (float)buf[i+1];
		float rbratio = (float)buf[i+2] / (float)buf[i];
		//if (rbratio > 0.5 && rbratio != rgratio) //If blue is sufficiently low and the ratios are not equal (white, grey or black)
		if (rbratio > 0.5) //If blue is sufficiently lower than red
		{
			// Note: These ratio values could be adjusted to widen or narrow selection
			if (rgratio > 0.75 && rgratio < 1.50)
				buf[i+2] *= 0.25;
			// Note: These two statements are to smooth the new color transitions (also, see note above)
			else if (rgratio > 0.50 && rgratio < 1.75)
				buf[i+2] *= 0.75;
			else if (rgratio > 0.25 && rgratio < 2.5)
				buf[i+2] *= 0.95;
		}
	}
}

int main(int argc, char *argv[])
{
	int comm_rank, comm_size;
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &comm_rank);
	MPI_Comm_size(MPI_COMM_WORLD, &comm_size);

	if (argc < 3)
	{
		if (comm_rank == COMM_ROOT)
			printerr("Usage: %s INFILE OUTFILE", argv[0]);
		else
			exit(EXIT_FAILURE);

	}

	// Open input image and get info
	IplImage *in;
	if ((in = cvLoadImage(argv[1], CV_LOAD_IMAGE_COLOR)) == NULL)
		printerr("Could not open input file %s: %s", argv[1], strerror(errno));

	/// Calculate our sizes and displacements
	int *size, *disp;
	if ((size = (int*)malloc(sizeof(int)*comm_size)) == NULL)
		printerr("[Rank %d] Could not allocate memory for buffer: %s", comm_rank, strerror(errno));
	if ((disp = (int*)malloc(sizeof(int)*comm_size)) == NULL)
		printerr("[Rank %d] Could not allocate memory for buffer: %s", comm_rank, strerror(errno));

	int size_per_proc = in->imageSize / comm_size;
	int rem = in->imageSize % comm_size;
	int sum = 0;
	for (int i = 0; i < comm_size; i++)
	{
		size[i] = size_per_proc;
		if (rem)
		{
			size[i]++;
			rem--;
		}

		disp[i] = sum;
		sum += size[i];
	}

	processImg(in->imageData + disp[comm_rank], size[comm_rank], in->nChannels);

	char *recvbuf;
	if (comm_rank == COMM_ROOT)
		recvbuf = (char*)malloc(in->imageSize);

	MPI_Gatherv(in->imageData + disp[comm_rank], size[comm_rank], MPI_CHAR, recvbuf, size, disp, MPI_CHAR, COMM_ROOT, MPI_COMM_WORLD);
	MPI_Barrier(MPI_COMM_WORLD);

	if (comm_rank == COMM_ROOT)
	{
		IplImage *out = cvCreateImageHeader(cvSize(in->width, in->height), in->depth, in->nChannels);
		if (!out)
			printerr("Could not allocate buffer for output file: %s", strerror(errno));
		out->imageData = recvbuf;
		if (!cvSaveImage(argv[2], out, NULL))
			printerr("Could not save output file %s: %s", argv[2], strerror(errno));
		cvReleaseImage(&out);
		free(recvbuf);
	}
	cvReleaseImage(&in);
	free(size);
	free(disp);

	MPI_Finalize();
	return 0;
}
