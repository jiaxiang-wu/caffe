#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/quan_inner_product_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void QuanInnerProductLayer<Dtype>::MatrixTranspose_cpu(
    Dtype* arr, int num_rows, int num_cols) {
  Dtype* buf = trans_buf_.mutable_cpu_data();
  caffe_copy(num_rows * num_cols, arr, buf);
  for (int idx_col = 0; idx_col < num_cols; idx_col++) {
    const Dtype* src = buf + idx_col;
    Dtype* dst = arr + idx_col * num_rows;
    for (int idx_row = 0; idx_row < num_rows; idx_row++) {
      dst[idx_row] = src[idx_row * num_cols];
    }
  }
}

template <typename Dtype>
void QuanInnerProductLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  // Extract hyper-parameters from the *.prototxt file
  num_word_  = this->layer_param_.quan_inner_product_param().num_word();
  len_word_  = this->layer_param_.quan_inner_product_param().len_word();
  N_         = this->layer_param_.quan_inner_product_param().num_output();
  bias_term_ = this->layer_param_.quan_inner_product_param().bias_term();
  const int axis = bottom[0]->CanonicalAxisIndex(
      this->layer_param_.quan_inner_product_param().axis());

  // Dimensions starting from "axis" are "flattened" into a single
  // length K_ vector. For example, if bottom[0]'s shape is (N, C, H, W),
  // and axis == 1, N inner products with dimension CHW are performed.
  K_ = bottom[0]->count(axis);
  num_scbk_ = (K_ - 1) / len_word_ + 1;

  // Check if we need to set up the weights
  if (this->blobs_.size() > 0) {
    LOG(INFO) << "Skipping parameter initialization";
  } else {
    // Pre-allocate blobs w.r.t the existence of bias term
    if (bias_term_) {
      this->blobs_.resize(3);
    } else {
      this->blobs_.resize(2);
    }

    // Initialize the set of sub-codebooks
    vector<int> scbk_shape(3);
    scbk_shape[0] = num_scbk_;
    scbk_shape[1] = num_word_;
    scbk_shape[2] = len_word_;
    this->blobs_[0].reset(new Blob<Dtype>(scbk_shape));
    // fill the set of sub-codebooks
    shared_ptr<Filler<Dtype> > weight_filler(GetFiller<Dtype>(
        this->layer_param_.quan_inner_product_param().weight_filler()));
    weight_filler->Fill(this->blobs_[0].get());

    // Initialize the set of quantization indicators
    vector<int> quan_ind_shape(2);
    quan_ind_shape[0] = num_scbk_;
    quan_ind_shape[1] = N_;
    this->blobs_[1].reset(new Blob<Dtype>(quan_ind_shape));
    // fill the set of quantization indicators
    Dtype* quan_ind = this->blobs_[1]->mutable_cpu_data();
    for (int idx = 0; idx < this->blobs_[1]->count(); idx++) {
      quan_ind[idx] = static_cast<Dtype>(rand() % num_word_);
    }

    // If necessary, intiialize and fill the bias term
    if (bias_term_) {
      vector<int> bias_shape(1, N_);
      this->blobs_[2].reset(new Blob<Dtype>(bias_shape));
      shared_ptr<Filler<Dtype> > bias_filler(GetFiller<Dtype>(
          this->layer_param_.quan_inner_product_param().bias_filler()));
      bias_filler->Fill(this->blobs_[2].get());
    }
  }  // parameter initialization

  // Specify whether the diff of each param blob should be computed
  this->param_propagate_down_.resize(this->blobs_.size(), true);
  this->param_propagate_down_[1] = false;  // skip for quantization indicators
}

template <typename Dtype>
void QuanInnerProductLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  // Figure out the dimensions
  const int axis = bottom[0]->CanonicalAxisIndex(
      this->layer_param_.inner_product_param().axis());
  const int new_K = bottom[0]->count(axis);
  CHECK_EQ(K_, new_K)
      << "Input size incompatible with inner product parameters.";

  // The first "axis" dimensions are independent inner products; the total
  // number of these is M_, the product over these dimensions.
  M_ = bottom[0]->count(0, axis);

  // The top shape will be the bottom shape with the flattened axes dropped,
  // and replaced by a single axis with dimension num_output (N_).
  vector<int> top_shape = bottom[0]->shape();
  top_shape.resize(axis + 1);
  top_shape[axis] = N_;
  top[0]->Reshape(top_shape);

  // Set up the bias multiplier
  if (bias_term_) {
    vector<int> bias_shape(1, M_);
    bias_multiplier_.Reshape(bias_shape);
    caffe_set(M_, (Dtype)1., bias_multiplier_.mutable_cpu_data());
  }

  // Create a look-up table to store pre-computed inner products
  vector<int> lkup_tbl_shape(3);
  lkup_tbl_shape[0] = 1;
  lkup_tbl_shape[1] = num_word_;
  lkup_tbl_shape[2] = M_;
  lkup_tbl_.Reshape(lkup_tbl_shape);

  // Create a memory buffer for the matrix transposition
  vector<int> trans_buf_shape(2);
  trans_buf_shape[0] = M_;
  trans_buf_shape[1] = (N_ > K_) ? N_ : K_;
  trans_buf_.Reshape(trans_buf_shape);
}

template <typename Dtype>
void QuanInnerProductLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  // Tranpose the input blob into the <D x N> shape
  MatrixTranspose_cpu(bottom[0]->mutable_cpu_data(), M_, K_);

  // SCHEME #1: for-loop accumulation
  // Compute the layer response, from <D_i x N> to <D_o x N>
  const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* scbk_sel = this->blobs_[0]->cpu_data();
  const Dtype* quan_ind_sel = this->blobs_[1]->cpu_data();
  Dtype* lkup_tbl_data = lkup_tbl_.mutable_cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  caffe_set(top[0]->count(), (Dtype)0., top[0]->mutable_cpu_data());
  for (int idx_scbk = 0; idx_scbk < num_scbk_; idx_scbk++) {
    // STAGE #1: inner product pre-computation
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_word_, M_, len_word_,
        (Dtype)1., scbk_sel, bottom_data, (Dtype)0., lkup_tbl_data);
    bottom_data += len_word_ * M_;
    scbk_sel += num_word_ * len_word_;

    // STAGE #2: approximate layer response computation
    for (int idx_output = 0; idx_output < N_; idx_output++) {
      int idx_word = static_cast<int>(quan_ind_sel[idx_output] + 0.5);
      caffe_axpy<Dtype>(M_, (Dtype)1.,
          lkup_tbl_data + idx_word * M_, top_data + idx_output * M_);
    }
    quan_ind_sel += N_;
  }

  // Tranpose input/output blobs into the <N x D> shape
  MatrixTranspose_cpu(bottom[0]->mutable_cpu_data(), K_, M_);
  MatrixTranspose_cpu(top[0]->mutable_cpu_data(), N_, M_);

  // If necessary, add the bias term
  if (bias_term_) {
    Dtype* top_data = top[0]->mutable_cpu_data();
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, N_, 1, (Dtype)1.,
        bias_multiplier_.cpu_data(),
        this->blobs_[2]->cpu_data(), (Dtype)1., top_data);
  }
}

template <typename Dtype>
void QuanInnerProductLayer<Dtype>::Backward_cpu(
    const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  // Tranpose input/output blobs into the <D x N> shape
  MatrixTranspose_cpu(bottom[0]->mutable_cpu_data(), M_, K_);
  MatrixTranspose_cpu(top[0]->mutable_cpu_diff(), M_, N_);

  // Compute the gradient signal for set of sub-codebooks and layer input
  const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* top_diff = top[0]->cpu_diff();
  const Dtype* scbk_data_sel = this->blobs_[0]->cpu_data();
  const Dtype* quan_ind_sel = this->blobs_[1]->cpu_data();
  Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
  Dtype* scbk_diff_sel = this->blobs_[0]->mutable_cpu_diff();
  Dtype* lkup_tbl_diff = lkup_tbl_.mutable_cpu_diff();
  for (int idx_scbk = 0; idx_scbk < num_scbk_; idx_scbk++) {
    // Compute the gradient signal of the look-up table
    caffe_set(lkup_tbl_.count(), (Dtype)0., lkup_tbl_.mutable_cpu_diff());
    for (int idx_output = 0; idx_output < N_; idx_output++) {
      int idx_word = static_cast<int>(quan_ind_sel[idx_output] + 0.5);
      caffe_axpy<Dtype>(M_, (Dtype)1.,
          top_diff + idx_output * M_, lkup_tbl_diff + idx_word * M_);
    }
    quan_ind_sel += N_;

    // Compute the gradient signal of the sub-codebook
    if (this->param_propagate_down_[0]) {
      caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, num_word_, len_word_, M_,
          (Dtype)1., lkup_tbl_diff, bottom_data, (Dtype)0., scbk_diff_sel);
    }
    bottom_data += len_word_ * M_;
    scbk_diff_sel += num_word_ * len_word_;

    // Compute the gradient signal of the layer input
    if (propagate_down[0]) {
      caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans, len_word_, M_, num_word_,
          (Dtype)1., scbk_data_sel, lkup_tbl_diff, (Dtype)0., bottom_diff);
    }
    bottom_diff += len_word_ * M_;
    scbk_data_sel += num_word_ * len_word_;
  }

  // Tranpose input/output blobs into the <N x D> shape
  MatrixTranspose_cpu(bottom[0]->mutable_cpu_data(), K_, M_);
  MatrixTranspose_cpu(bottom[0]->mutable_cpu_diff(), K_, M_);
  MatrixTranspose_cpu(top[0]->mutable_cpu_diff(), N_, M_);

  // If necessary, compute the gradient signal of the bias term
  if (bias_term_ && this->param_propagate_down_[2]) {
    const Dtype* top_diff = top[0]->cpu_diff();
    caffe_cpu_gemv<Dtype>(CblasTrans, M_, N_, (Dtype)1., top_diff,
        bias_multiplier_.cpu_data(), (Dtype)1.,
        this->blobs_[2]->mutable_cpu_diff());
  }
}

#ifdef CPU_ONLY
STUB_GPU(QuanInnerProductLayer);
#endif

INSTANTIATE_CLASS(QuanInnerProductLayer);
REGISTER_LAYER_CLASS(QuanInnerProduct);

}  // namespace caffe
