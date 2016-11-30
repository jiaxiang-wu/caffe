#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/quan_inner_product_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
__global__ void MatrixTranspose_gpu_kernel(const int n,
    Dtype* arr, const Dtype* buf, const int num_rows, const int num_cols) {
  CUDA_KERNEL_LOOP(index, n) {
    int idx_col = index;
    const Dtype* src = buf + idx_col;
    Dtype* dst = arr + idx_col * num_rows;
    for (int idx_row = 0; idx_row < num_rows; idx_row++) {
      dst[idx_row] = src[idx_row * num_cols];
    }
  }
}

template <typename Dtype>
void QuanInnerProductLayer<Dtype>::MatrixTranspose_gpu(
    Dtype* arr, const int num_rows, const int num_cols) {
  // Copy the original data into the memory buffer
  Dtype* buf = trans_buf_.mutable_gpu_data();
  caffe_copy(num_rows * num_cols, arr, buf);

  // Re-organize the data using the memory buffer
  int num_kernels = num_cols;
  MatrixTranspose_gpu_kernel<Dtype><<<CAFFE_GET_BLOCKS(num_kernels),
                                      CAFFE_CUDA_NUM_THREADS>>>(
      num_kernels, arr, buf, num_rows, num_cols);
  CUDA_POST_KERNEL_CHECK;
}

template <typename Dtype>
__global__ void Forward_gpu_kernel(const int n, const Dtype* lkup_tbl, 
    const int* quan_ind_sel, Dtype* top_data, const int M_) {
  CUDA_KERNEL_LOOP(index, n) {
    int idx_output = index;
    int idx_word = quan_ind_sel[idx_output];
    const Dtype* lkup_tbl_sel = lkup_tbl + idx_word * M_;
    Dtype* top_data_sel = top_data + idx_output * M_;
    for (int m = 0; m < M_; m++) {
      top_data_sel[m] += lkup_tbl_sel[m];
    }
  }
}

template <typename Dtype>
__global__ void Forward_gpu_kernel_s1(const int num_scbk, 
    const int num_word, const int len_word, const int num_smpl, 
    const Dtype* bottom_data, const Dtype* scbk_mat, Dtype* lkup_tbl) {
  CUDA_KERNEL_LOOP(idx_scbk, num_scbk) {
    // Move pointers to the current subspace
    const Dtype* btm_data = bottom_data + idx_scbk * (len_word * num_smpl);
    const Dtype* scbk_sel = scbk_mat + idx_scbk * (num_word * len_word);
    Dtype* lkup_tbl_sel = lkup_tbl + idx_scbk * (num_word * num_smpl);

    // Inner product pre-computation
    for (int idx_word = 0; idx_word < num_word; idx_word++) {
      const Dtype* p1 = scbk_sel + idx_word * len_word;
      Dtype* q = lkup_tbl_sel + idx_word * num_smpl;
      for (int dim_word = 0; dim_word < len_word; dim_word++) {
        const Dtype mul = p1[dim_word];
        const Dtype* p2 = btm_data + dim_word * num_smpl;
        for (int idx_smpl = 0; idx_smpl < num_smpl; idx_smpl++) {
          q[idx_smpl] += mul * p2[idx_smpl];
        }
      }
    }
  }
}


template <typename Dtype>
__global__ void Forward_gpu_kernel_s2(const int num_output,
    const int num_scbk, const int num_word, const int num_smpl,
    const Dtype* lkup_tbl, const int* quan_ind, Dtype* top_data) {
  CUDA_KERNEL_LOOP(idx_output, num_output) {
    // Move pointer to the current output node
    const int* quan_ind_sel = quan_ind + idx_output;
    Dtype* top_data_sel = top_data + idx_output * num_smpl;

    // Approximate layer response computation
    for (int idx_scbk = 0; idx_scbk < num_scbk; idx_scbk++) {
      const int idx_word = *quan_ind_sel;
      const Dtype* lkup_tbl_sel = lkup_tbl 
          + (idx_scbk * num_word + idx_word) * num_smpl;
      for (int idx_smpl = 0; idx_smpl < num_smpl; idx_smpl++) {
        top_data_sel[idx_smpl] += lkup_tbl_sel[idx_smpl];
      }
      quan_ind_sel += num_output;
    }
  }
}

template <typename Dtype>
__global__ void Forward_gpu_kernel_fn(const int num_smpl, const int num_output,
    const int num_scbk, const int num_word, const int len_word,
    const Dtype* btm_data, const Dtype* scbk_mat, const int* quan_ind,
    Dtype* top_data) {
  CUDA_KERNEL_LOOP(idx_smpl, num_smpl) {
    const int num_input = num_scbk * len_word;
    const Dtype* btm_data_sel = btm_data + idx_smpl * num_input;
    Dtype* top_data_sel = top_data + idx_smpl * num_output;
    Dtype* lkup_tbl = (Dtype*)malloc(num_word * sizeof(Dtype));
    Dtype* btm_data_buf = (Dtype*)malloc(len_word * sizeof(Dtype));
    Dtype* top_data_buf = (Dtype*)malloc(num_output * sizeof(Dtype));

    for (int idx_output = 0; idx_output < num_output; idx_output++) {
      top_data_buf[idx_output] = (Dtype)0.;
    }

    for (int idx_scbk = 0; idx_scbk < num_scbk; idx_scbk++) {
      // STAGE #1: inner product pre-computation
      for (int dim_word = 0; dim_word < len_word; dim_word++) {
        btm_data_buf[dim_word] = btm_data_sel[dim_word];
      }
      for (int idx_word = 0; idx_word < num_word; idx_word++) {
        Dtype val = (Dtype)0.;
        const Dtype* p = scbk_mat + idx_word * len_word;
        for (int dim_word = 0; dim_word < len_word; dim_word++) {
          val += btm_data_buf[dim_word] * p[dim_word];
        }
        lkup_tbl[idx_word] = val;
      }

      // STAGE #2: approximate layer response computation
      for (int idx_output = 0; idx_output < num_output; idx_output++) {
        int idx_word = quan_ind[idx_output];
        top_data_buf[idx_output] += lkup_tbl[idx_word];
      }

      // Post-processing
      btm_data_sel += len_word;
      scbk_mat += num_word * len_word;
      quan_ind += num_output;
    }

    for (int idx_output = 0; idx_output < num_output; idx_output++) {
      top_data_sel[idx_output] = top_data_buf[idx_output];
    }

    free(lkup_tbl);
    free(btm_data_buf);
    free(top_data_buf);
  }
}

template <typename Dtype>
void QuanInnerProductLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  // Tranpose the input blob into the <D x N> shape
  MatrixTranspose_gpu(bottom[0]->mutable_gpu_data(), M_, K_);

  // SCHEME #1: cuda kernels as the inner loop
  // Compute the layer response, from <D_i x N> to <D_o x N>
  const Dtype* bottom_data = bottom[0]->gpu_data();
  const Dtype* scbk_sel = this->blobs_[0]->gpu_data();
  const int* quan_ind_sel = (int*)(this->blobs_[1]->gpu_data());
  Dtype* top_data = top[0]->mutable_gpu_data();
  caffe_gpu_set(top[0]->count(), (Dtype)0., top[0]->mutable_gpu_data());
  for (int idx_scbk = 0; idx_scbk < num_scbk_; idx_scbk++) {
    // STAGE #1: inner product pre-computation
    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, 
        num_word_, M_, len_word_, (Dtype)1., scbk_sel, bottom_data,
        (Dtype)0., lkup_tbl_.mutable_gpu_data());
    bottom_data += len_word_ * M_;
    scbk_sel += num_word_ * len_word_;

    // STAGE #2: approximate layer response computation
    int num_kernels = N_;
    Forward_gpu_kernel<Dtype><<<CAFFE_GET_BLOCKS(num_kernels),
                                CAFFE_CUDA_NUM_THREADS>>>(
        num_kernels, lkup_tbl_.gpu_data(), quan_ind_sel, top_data, M_);
    CUDA_POST_KERNEL_CHECK;
    quan_ind_sel += N_;
  }

  /*
  // SCHEME #2: cuda kernels as the outer loop
  // STAGE #1: inner product pre-computation
  int num_kernels_s1 = num_scbk_;
  caffe_gpu_set<Dtype>(
      lkup_tbl_.count(), (Dtype)0., lkup_tbl_.mutable_gpu_data());
  Forward_gpu_kernel_s1<Dtype><<<CAFFE_GET_BLOCKS(num_kernels_s1),
                                 CAFFE_CUDA_NUM_THREADS>>>(
      num_scbk_, num_word_, len_word_, M_, bottom[0]->gpu_data(), 
      this->blobs_[0]->gpu_data(), lkup_tbl_.mutable_gpu_data());
  CUDA_POST_KERNEL_CHECK;

  // STAGE #2: approximate layer response computation
  int num_kernels_s2 = N_;
  caffe_gpu_set<Dtype>(
      top[0]->count(), (Dtype)0., top[0]->mutable_gpu_data());
  Forward_gpu_kernel_s2<Dtype><<<CAFFE_GET_BLOCKS(num_kernels_s2),
                                 CAFFE_CUDA_NUM_THREADS>>>(
      N_, num_scbk_, num_word_, M_, lkup_tbl_.gpu_data(),
      reinterpret_cast<const int*>(this->blobs_[1]->gpu_data()), 
      top[0]->mutable_gpu_data());
  CUDA_POST_KERNEL_CHECK;
  */

  /*
  // SCHEME #3: cuda kernels as the outer loop
  int num_kernels = M_;
  caffe_gpu_set<Dtype>(
      top[0]->count(), (Dtype)0., top[0]->mutable_cpu_data());
  Forward_gpu_kernel_fn<Dtype><<<CAFFE_GET_BLOCKS(num_kernels),
                                 CAFFE_CUDA_NUM_THREADS>>>(
      M_, N_, num_scbk_, num_word_, len_word_, 
      bottom[0]->gpu_data(), this->blobs_[0]->gpu_data(),
      reinterpret_cast<const int*>(this->blobs_[1]->gpu_data()),
      top[0]->mutable_cpu_data());
  CUDA_POST_KERNEL_CHECK;
  */

  // Tranpose input/output blobs into the <N x D> shape
  MatrixTranspose_gpu(bottom[0]->mutable_gpu_data(), K_, M_);
  MatrixTranspose_gpu(top[0]->mutable_gpu_data(), N_, M_);

  // If necessary, add the bias term
  if (bias_term_) {
    Dtype* top_data = top[0]->mutable_gpu_data();
    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, N_, 1, (Dtype)1.,
        bias_multiplier_.gpu_data(),
        this->blobs_[2]->gpu_data(), (Dtype)1., top_data);
  }
}

template <typename Dtype>
__global__ void Backward_gpu_kernel(const int n, const Dtype* top_diff, 
    const int* quan_ind_sel, Dtype* lkup_tbl, const int N_, const int M_) {
  CUDA_KERNEL_LOOP(index, n) {
    int idx_word = index;
    const Dtype* top_diff_sel = top_diff;
    Dtype* lkup_tbl_sel = lkup_tbl + idx_word * M_;
    for (int idx_output = 0; idx_output < N_; idx_output++) {
      if (quan_ind_sel[idx_output] == idx_word) {
        for (int m = 0; m < M_; m++) {
          lkup_tbl_sel[m] += top_diff_sel[m];
        }
      }
      top_diff_sel += M_;
    }
  }
}

template <typename Dtype>
void QuanInnerProductLayer<Dtype>::Backward_gpu(
    const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  // Tranpose input/output blobs into the <D x N> shape
  MatrixTranspose_gpu(bottom[0]->mutable_gpu_data(), M_, K_);
  MatrixTranspose_gpu(top[0]->mutable_gpu_diff(), M_, N_);

  // Compute the gradient signal for set of sub-codebooks and layer input
  const Dtype* bottom_data = bottom[0]->gpu_data();
  const Dtype* top_diff = top[0]->gpu_diff();
  const Dtype* scbk_data_sel = this->blobs_[0]->gpu_data();
  const int* quan_ind_sel = (int*)(this->blobs_[1]->gpu_data());
  Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
  Dtype* scbk_diff_sel = this->blobs_[0]->mutable_gpu_diff();
  for (int idx_scbk = 0; idx_scbk < num_scbk_; idx_scbk++) {
    // Compute the gradient signal of the look-up table
    caffe_gpu_set(lkup_tbl_.count(), (Dtype)0., lkup_tbl_.mutable_gpu_diff());
    int num_kernels = num_word_;
    Backward_gpu_kernel<Dtype><<<CAFFE_GET_BLOCKS(num_kernels),
                                 CAFFE_CUDA_NUM_THREADS>>>(
        num_kernels, top_diff, quan_ind_sel, 
        lkup_tbl_.mutable_gpu_diff(), N_, M_);
    CUDA_POST_KERNEL_CHECK;
    /*
    for (int idx_output = 0; idx_output < N_; idx_output++) {
      int idx_word = quan_ind_sel[idx_output];
      caffe_gpu_axpy<Dtype>(M_, (Dtype)1., top_diff + idx_output * M_,
          lkup_tbl_.mutable_gpu_diff() + idx_word * M_);
    }
    */
    quan_ind_sel += N_;

    // Compute the gradient signal of the sub-codebook
    if (this->param_propagate_down_[0]) {
      caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans, 
          num_word_, len_word_, M_, (Dtype)1.,
          lkup_tbl_.gpu_diff(), bottom_data, (Dtype)0., scbk_diff_sel);
    }
    bottom_data += len_word_ * M_;
    scbk_diff_sel += num_word_ * len_word_;

    // Compute the gradient signal of the layer input
    if (propagate_down[0]) {
      caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans,
          len_word_, M_, num_word_, (Dtype)1.,
          scbk_data_sel, lkup_tbl_.gpu_diff(), (Dtype)0., bottom_diff);
    }
    bottom_diff += len_word_ * M_;
    scbk_data_sel += num_word_ * len_word_;
  }

  // Tranpose input/output blobs into the <N x D> shape
  MatrixTranspose_gpu(bottom[0]->mutable_gpu_data(), K_, M_);
  MatrixTranspose_gpu(bottom[0]->mutable_gpu_diff(), K_, M_);
  MatrixTranspose_gpu(top[0]->mutable_gpu_diff(), N_, M_);

  // If necessary, compute the gradient signal of the bias term
  if (bias_term_ && this->param_propagate_down_[2]) {
    const Dtype* top_diff = top[0]->gpu_diff();
    caffe_gpu_gemv<Dtype>(CblasTrans, M_, N_, (Dtype)1., top_diff,
        bias_multiplier_.gpu_data(), (Dtype)1.,
        this->blobs_[2]->mutable_gpu_diff());
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(QuanInnerProductLayer);

}  // namespace caffe
