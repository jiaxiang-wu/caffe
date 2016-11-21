#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/quan_inner_product_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void MatrixTranspose(Dtype* arr, int num_rows, int num_cols) {
  vector<bool> marker(num_rows * num_cols, false);
  int mod_factor = num_rows * num_cols - 1;
  for (int idx = 1; idx < mod_factor; ) {  // skip the first & last elements
    int idxBegn = idx;
    marker[idxBegn] = true;
    while (true) {
      int idxNext = (idx * num_cols) % mod_factor;
      marker[idxNext] = true;
      if (idxNext == idxBegn) {
        break;
      } else {
        Dtype valTemp = arr[idx];
        arr[idx] = arr[idxNext];
        arr[idxNext] = valTemp;
        idx = idxNext;
      }
    }
    for (idx = idxBegn + 1; idx < mod_factor && marker[idx]; idx++) ;
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
    int* quan_ind = (int*)(this->blobs_[1]->mutable_cpu_data());
    for (int idx = 0; idx < this->blobs_[1]->count(); idx++) {
      quan_ind[idx] = rand() % num_word_;
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
  vector<int> lkup_tbl_shape(2);
  lkup_tbl_shape[0] = num_word_;
  lkup_tbl_shape[1] = M_;
  lkup_tbl_.Reshape(lkup_tbl_shape);
}

template <typename Dtype>
void QuanInnerProductLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  // Tranpose the input blob to the shape of <D_i x N>
  MatrixTranspose(bottom[0]->mutable_cpu_data(), M_, K_);

  // Compute the layer response, from <D_i x N> to <D_o x N>
  const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* scbk_sel = this->blobs_[0]->cpu_data();
  const int* quan_ind_sel = (int*)(this->blobs_[1]->cpu_data());
  Dtype* top_data = top[0]->mutable_cpu_data();
  caffe_set(top[0]->count(), (Dtype)0., top[0]->mutable_cpu_data());
  for (int idx_scbk = 0; idx_scbk < num_scbk_; idx_scbk++) {
    // STAGE #1: inner product pre-computation
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, 
        num_word_, M_, len_word_, (Dtype)1., scbk_sel, bottom_data,
        (Dtype)0., lkup_tbl_.mutable_cpu_data());
    bottom_data += len_word_ * M_;
    scbk_sel += num_word_ * len_word_;

    // STAGE #2: approximate layer response computation
    for (int idx_output = 0; idx_output < N_; idx_output++) {
      int idx_word = quan_ind_sel[idx_output];
      caffe_axpy<Dtype>(M_, (Dtype)1., 
          lkup_tbl_.cpu_data() + idx_word * M_, top_data + idx_output * M_);
    }
    quan_ind_sel += N_;
  }

  // Tranpose the output blob to the shape of <N x D_o>
  MatrixTranspose(bottom[0]->mutable_cpu_data(), K_, M_);
  MatrixTranspose(top[0]->mutable_cpu_data(), N_, M_);

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
  // Reset the gradient signal of the set of codebooks and the layer input
  if (this->param_propagate_down_[0]) {
    caffe_set(this->blobs_[0]->count(), 
        (Dtype)0., this->blobs_[0]->mutable_cpu_diff());
  }
  if (propagate_down[0]) {
    caffe_set(bottom[0]->count(), (Dtype)0., bottom[0]->mutable_cpu_diff());
  }

  // Compute the gradient signal for one instance at a time
  Dtype lkup_tbl_vec[num_word_];
  for (int m = 0; m < M_; m++) {
    // Compute the gradient signal for one subspace at a time
    const Dtype* bottom_data = bottom[0]->cpu_data() + m * K_;
    const Dtype* top_diff = top[0]->cpu_diff() + m * N_;
    const Dtype* scbk_data_sel = this->blobs_[0]->cpu_data();
    const int* quan_ind_vec = (int*)(this->blobs_[1]->cpu_data());
    Dtype* scbk_diff_sel = this->blobs_[0]->mutable_cpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff() + m * K_;
    for (int idx_scbk = 0; idx_scbk < num_scbk_; idx_scbk++) {
      // Compute the gradient signal of the look-up table
      caffe_set(num_word_, (Dtype)0., lkup_tbl_vec);
      for (int idx_output = 0; idx_output < N_; idx_output++) {
        lkup_tbl_vec[quan_ind_vec[idx_output]] += top_diff[idx_output];
      }

      // Compute the gradient signal of the sub-codebook
      if (this->param_propagate_down_[0]) {
        caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_word_, len_word_,
            1, (Dtype)1., lkup_tbl_vec, bottom_data, (Dtype)1., scbk_diff_sel);
      }

      // Compute the gradient signal of the layer input
      if (propagate_down[0]) {
        caffe_cpu_gemv<Dtype>(CblasTrans, num_word_, len_word_, (Dtype)1.0,
            scbk_data_sel, lkup_tbl_vec, (Dtype)0., bottom_diff);
      }

      // move pointers to the next subspace
      bottom_data += len_word_;
      scbk_data_sel += num_word_ * len_word_;
      quan_ind_vec += N_;
      scbk_diff_sel += num_word_ * len_word_;
      bottom_diff += len_word_;
    }
  }

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
