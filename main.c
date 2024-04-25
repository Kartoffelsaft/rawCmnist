#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <assert.h>
#include <gsl/gsl_linalg.h>
#define STB_DS_IMPLEMENTATION
#include <stb/stb_ds.h>
#include <math.h>

double relu(double x) {
    return x > 0? x:x*0.0078125;
}

double drelu(double x) {
    return x > 0? 1:0.0078125;
}

void softmax(gsl_vector* v) {
    for (int i = 0; i < v->size; i++) {
        gsl_vector_set(v, i, exp(gsl_vector_get(v, i)));
    }
    double sum = gsl_vector_sum(v);
    gsl_blas_dscal(1.0/sum, v);
}

void map_matrix_elements(gsl_matrix* mtx, double (*mapping)(double)) {
    for (int i = 0; i < mtx->size1; i++) for (int j = 0; j < mtx->size2; j++) {
        gsl_matrix_set(mtx, i, j, mapping(gsl_matrix_get(mtx, i, j)));
    }
}

typedef struct {
    gsl_matrix** weights;
    gsl_vector** biases;
} Model;

void initializeModel(Model* to, size_t* shape) {
    if (shape == NULL) return;
    if (shape[0] == 0) return;
    if (shape[1] == 0) return;

    to->weights = NULL;
    to->biases = NULL;

    for (int i = 1; shape[i] != 0; i++) {
        arrpush(to->weights, gsl_matrix_alloc(shape[i], shape[i-1]));
        arrpush(to->biases, gsl_vector_alloc(shape[i]));
    }

    for (int i = 0; i < arrlen(to->weights); i++) {
        for (int r = 0; r < to->weights[i]->size1; r++) for (int c = 0; c < to->weights[i]->size2; c++) {
            gsl_matrix_set(to->weights[i], r, c, (rand() / (float)RAND_MAX) - 0.5);
        }
        for (int j = 0; j < to->biases[i]->size; j++) {
            gsl_vector_set(to->biases[i], j, (rand() / (float)RAND_MAX) - 0.5);
        }
    }
}

gsl_matrix** feedForward(Model* m, gsl_matrix* input) {
    gsl_matrix** outputs = NULL;
    for (int i = 0; i < arrlen(m->weights); i++) {
        arrpush(outputs, gsl_matrix_alloc(m->weights[i]->size1, input->size2));
    }

    // fenceposting, using input as "previous" output
    gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, m->weights[0], input, 0.0, outputs[0]);

    for (int i = 0; i < outputs[0]->size2; i++) {
        gsl_vector_view column = gsl_matrix_column(outputs[0], i);
        gsl_blas_daxpy(1.0, m->biases[0], &column.vector);
    }
    for (int i = 0; i < outputs[0]->size1; i++) for (int j = 0; j < outputs[0]->size2; j++) {
        gsl_matrix_set(outputs[0], i, j, relu(gsl_matrix_get(outputs[0], i, j)));
    }

    for (int l = 1; l < arrlen(outputs); l++) {
        gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, m->weights[l], outputs[l-1], 0.0, outputs[l]);

        for (int i = 0; i < outputs[l]->size2; i++) {
            gsl_vector_view column = gsl_matrix_column(outputs[l], i);
            gsl_blas_daxpy(1.0, m->biases[l], &column.vector);
        }

        if (l != arrlen(outputs)-1) { // last layer uses softmax for non-linearization
            map_matrix_elements(outputs[l], relu);
        }
    }

    gsl_vector_view outputExample = gsl_matrix_column(arrlast(outputs), 0);

    for (int c = 0; c < arrlast(outputs)->size2; c++) {
        gsl_vector_view col = gsl_matrix_column(arrlast(outputs), c);
        softmax(&col.vector);
    }

    return outputs;
}

void feedBackward(Model* m, gsl_matrix* input, gsl_matrix* expectedOutput, gsl_matrix** realOutputs, double alpha) {
    gsl_matrix_sub(arrlast(realOutputs), expectedOutput);

    gsl_matrix** dweights = NULL;
    gsl_vector** dbiases = NULL;

    for (int i = 0; i < arrlen(m->weights); i++) {
        arrpush(dweights, gsl_matrix_alloc(m->weights[i]->size1, m->weights[i]->size2));
    }
    for (int i = 0; i < arrlen(m->biases); i++) {
        arrpush(dbiases, gsl_vector_alloc(m->biases[i]->size));
    }

    for (int i = arrlen(dweights) - 1; i > 0; i--) {
        for (int j = 0; j < dbiases[i]->size; j++) {
            gsl_vector_view dzView = gsl_matrix_row(realOutputs[i], j);
            gsl_vector_set(dbiases[i], j, gsl_vector_sum(&dzView.vector));
        }
        gsl_blas_dscal(1.0/expectedOutput->size2, dbiases[i]);

        gsl_blas_dgemm(CblasNoTrans, CblasTrans, 1.0/expectedOutput->size2, realOutputs[i], realOutputs[i-1], 0.0, dweights[i]);

        gsl_matrix* drelus = gsl_matrix_alloc(realOutputs[i-1]->size1, realOutputs[i-1]->size2);
        gsl_matrix_memcpy(drelus, realOutputs[i-1]);
        map_matrix_elements(drelus, drelu);

        gsl_blas_dgemm(CblasTrans, CblasNoTrans, 1.0, m->weights[i], realOutputs[i], 0.0, realOutputs[i-1]);
        gsl_matrix_mul_elements(realOutputs[i-1], drelus);

        gsl_matrix_free(drelus);
    }
    { // fenceposting
        for (int j = 0; j < dbiases[0]->size; j++) {
            gsl_vector_view dzView = gsl_matrix_row(realOutputs[0], j);
            gsl_vector_set(dbiases[0], j, gsl_vector_sum(&dzView.vector));
        }
        gsl_blas_dscal(1.0/expectedOutput->size2, dbiases[0]);

        gsl_blas_dgemm(CblasNoTrans, CblasTrans, 1.0/expectedOutput->size2, realOutputs[0], input, 0.0, dweights[0]);
    }

    for (int i = 0; i < arrlen(dweights); i++) {
        gsl_matrix_scale(dweights[i], alpha);
        gsl_matrix_sub(m->weights[i], dweights[i]);

        gsl_vector_scale(dbiases[i], alpha);
        gsl_vector_sub(m->biases[i], dbiases[i]);
    }

    // free derivative matricies
    for (int i = 0; i < arrlen(dweights); i++) {
        gsl_matrix_free(dweights[i]);
    }
    arrfree(dweights);
    for (int i = 0; i < arrlen(dbiases); i++) {
        gsl_vector_free(dbiases[i]);
    }
    arrfree(dbiases);
}

int main() {
    size_t imageCount = 0;
    size_t imageDataSize;
    double* imageData;
    uint8_t* imageLabels;

    {
        FILE* trainingImageFile = fopen("train-images.idx3-ubyte", "r");
        assert(getc(trainingImageFile) == 0x00);
        assert(getc(trainingImageFile) == 0x00);
        assert(getc(trainingImageFile) == 0x08);
        assert(getc(trainingImageFile) == 0x03);

        imageCount |= getc(trainingImageFile) << 24;
        imageCount |= getc(trainingImageFile) << 16;
        imageCount |= getc(trainingImageFile) <<  8;
        imageCount |= getc(trainingImageFile) <<  0;

        assert(getc(trainingImageFile) == 0x00);
        assert(getc(trainingImageFile) == 0x00);
        assert(getc(trainingImageFile) == 0x00);
        assert(getc(trainingImageFile) == 0x1c);

        assert(getc(trainingImageFile) == 0x00);
        assert(getc(trainingImageFile) == 0x00);
        assert(getc(trainingImageFile) == 0x00);
        assert(getc(trainingImageFile) == 0x1c);

        imageDataSize = imageCount * 0x1c * 0x1c;
        imageData = malloc(sizeof(double) * imageDataSize);

        for (int i = 0; i < imageDataSize; i++) {
            imageData[i] = getc(trainingImageFile) / 256.0;
        }

        fclose(trainingImageFile);


        FILE* trainingLabelFile = fopen("train-labels.idx1-ubyte", "r");

        assert(getc(trainingLabelFile) == 0x00);
        assert(getc(trainingLabelFile) == 0x00);
        assert(getc(trainingLabelFile) == 0x08);
        assert(getc(trainingLabelFile) == 0x01);

        size_t labelCount = 0;
        labelCount |= getc(trainingLabelFile) << 24;
        labelCount |= getc(trainingLabelFile) << 16;
        labelCount |= getc(trainingLabelFile) <<  8;
        labelCount |= getc(trainingLabelFile) <<  0;
        assert(labelCount == imageCount);

        imageLabels = malloc(labelCount);
        for (int i = 0; i < labelCount; i++) {
            imageLabels[i] = getc(trainingLabelFile);
        }

        fclose(trainingLabelFile);
    }

    // TEMPORARY: makes testing much faster if we use less of the data
    //imageCount = 100;

    gsl_matrix_view imagesMatrixView = gsl_matrix_view_array(imageData, imageCount, 28*28);

    gsl_matrix* idealOutput = gsl_matrix_alloc(10, imageCount);
    gsl_matrix_set_zero(idealOutput);
    for (int i = 0; i < imageCount; i++) {
        gsl_matrix_set(idealOutput, imageLabels[i], i, 1.0);
    }


    //////////// Initialize network
    Model model;
    initializeModel(&model, (size_t[]){28*28, 128, 32, 32, 10, 0});

    for (int iterations = 0; iterations < 1000; iterations++) {

        gsl_matrix* imageBatch = gsl_matrix_alloc(imagesMatrixView.matrix.size2, 200);
        size_t viewBegin = rand()%(imageCount-200);
        gsl_matrix_view imageBatchView = gsl_matrix_submatrix(&imagesMatrixView.matrix, viewBegin, 0, 200, imagesMatrixView.matrix.size2);
        gsl_matrix_view imageBatchIdealOutputView = gsl_matrix_submatrix(idealOutput, 0, viewBegin, 10, 200);
        gsl_matrix_transpose_memcpy(imageBatch, &imageBatchView.matrix);

        gsl_matrix** outputs = feedForward(&model, imageBatch);

        if (iterations%20 == 0) {
            printf("\nIterations: %d\n", iterations);

            for (int i = 0; i < 28*28; i++) {
                if (gsl_matrix_get(&imageBatchView.matrix, 0, i) > 0.5) {
                    printf("#");
                } else {
                    printf(" ");
                }

                if (i%28 == 27) {
                    printf("\n");
                }
            }
            printf("Predictions: \n");
            for (int i = 0; i < 10; i++) {
                printf("%d: %3.2f\n", i,
                    gsl_matrix_get(arrlast(outputs), i, 0)
                );
            }
        }

        feedBackward(&model, imageBatch, &imageBatchIdealOutputView.matrix, outputs, 0.2);

        gsl_matrix_free(imageBatch);

        for (int i = 0; i < arrlen(outputs); i++) {
            gsl_matrix_free(outputs[i]);
        }
        arrfree(outputs);
    }

    return 0;
}
