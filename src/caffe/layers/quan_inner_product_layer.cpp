#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/quan_inner_product_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

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
    Dtype* quan_ind_vec = this->blobs_[1]->mutable_cpu_data();
    for (int idx = 0; idx < this->blobs_[1]->count(); idx++) {
      quan_ind_vec[idx] = rand() % num_word_;
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

  // Allocate memory for the look-up table of inner products
  vector<int> lkup_tbl_shape(2);
  lkup_tbl_shape[0] = num_word_;
  lkup_tbl_shape[1] = num_scbk_;
  lkup_tbl_.Reshape(lkup_tbl_shape);
  
  // Compute the set of subspace indices for each (c_t, k) pair
  scbk_idxs_ = vector<vector<vector<int> > >(
      N_, vector<vector<int> >(num_word_, vector<int>(0)));
  for (int idx_scbk = 0; idx_scbk < num_scbk_; idx_scbk++) {
    const Dtype* quan_ind_vec = this->blobs_[1]->cpu_data() + idx_scbk * N_;
    for (int idx_output = 0; idx_output < N_; idx_output++) {
      scbk_idxs_[idx_output][quan_ind_vec[idx_output]].push_back(idx_scbk);
    }
  }
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
}

template <typename Dtype>
void QuanInnerProductLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  // Compute the layer response for one instance at a time
  Dtype lkup_tbl_vec[num_word_];
  for (int m = 0; m < M_; m++) {
    // STAGE #1: inner product pre-computation
    const Dtype* bottom_data_sel = bottom[0]->cpu_data() + m * K_;
    const Dtype* scbk_sel = this->blobs_[0]->cpu_data();
    for (int idx_scbk = 0; idx_scbk < num_scbk_; idx_scbk++) {
      // Compute inner products with all codewords in the sub-codebook
      caffe_cpu_gemv<Dtype>(CblasNoTrans, num_word_, len_word_, 
          (Dtype)1., scbk_sel, bottom_data_sel, (Dtype)0., lkup_tbl_vec);
      bottom_data_sel += len_word_;
      scbk_sel += num_word_ * len_word_;

      // Re-arrange inner products in the look-up table
      Dtype* lkup_tbl_sel = lkup_tbl_.mutable_cpu_data() + idx_scbk;
      for (int idx_word = 0; idx_word < num_word_; idx_word++) {
        *lkup_tbl_sel = lkup_tbl_vec[idx_word];
        lkup_tbl_sel += num_scbk_;
      }
    }
  
    // STAGE #2: approximate layer response computation
    Dtype* top_data = top[0]->mutable_cpu_data() + m * N_;
    for (int idx_output = 0; idx_output < N_; idx_output++) {
      Dtype val = (Dtype)0.;
      const Dtype* lkup_tbl_sel = lkup_tbl_.cpu_data();
      for (int idx_word = 0; idx_word < num_word_; idx_word++) {
        const vector<int>& scbk_idxs_sel = scbk_idxs_[idx_output][idx_word];
        for (std::size_t idx = 0; idx < scbk_idxs_sel.size(); idx++) {
          val += lkup_tbl_sel[scbk_idxs_sel[idx]];
        }
        lkup_tbl_sel += num_scbk_;
      }
      top_data[idx_output] = val;
    }
  }

  // If necessary, add the bias term
  Dtype* top_data = top[0]->mutable_cpu_data();
  if (bias_term_) {
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, N_, 1, (Dtype)1.,
        bias_multiplier_.cpu_data(),
        this->blobs_[2]->cpu_data(), (Dtype)1., top_data);
  }
}

template <typename Dtype>
void QuanInnerProductLayer<Dtype>::Backward_cpu(
    const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  // Compute the gradient signal of the set of sub-codebooks
  Dtype lkup_tbl_vec[num_word_];
  if (this->param_propagate_down_[0]) {
    // Reset the gradient signal of the set of sub-codebooks
    caffe_set(this->blobs_[0]->count(), 
        (Dtype)0., this->blobs_[0]->mutable_cpu_diff());

    // Compute the gradient signal for one instance at a time
    for (int m = 0; m < M_; m++) {
      // Compute the gradient signal of the look-up table
      caffe_set(lkup_tbl_.count(), (Dtype)0., lkup_tbl_.mutable_cpu_diff());
      const Dtype* top_diff = top[0]->cpu_diff() + m * N_;
      for (int idx_output = 0; idx_output < N_; idx_output++) {
        Dtype val = top_diff[idx_output];
        Dtype* lkup_tbl_sel = lkup_tbl_.mutable_cpu_diff();
        for (int idx_word = 0; idx_word < num_word_; idx_word++) {
          const vector<int>& scbk_idxs_sel = scbk_idxs_[idx_output][idx_word];
          for (std::size_t idx = 0; idx < scbk_idxs_sel.size(); idx++) {
            lkup_tbl_sel[scbk_idxs_sel[idx]] += val;
          }
        }
        lkup_tbl_sel += num_scbk_;
      }

      // Compute the gradient signal of the set of sub-codebooks
      const Dtype* bottom_data = bottom[0]->cpu_data() + m * K_;
      Dtype* scbk_sel = this->blobs_[0]->mutable_cpu_diff();
      for (int idx_scbk = 0; idx_scbk < num_scbk_; idx_scbk++) {
        // Re-arrange the gradient signal of the look-up table
        const Dtype* lkup_tbl_sel = lkup_tbl_.cpu_diff() + idx_scbk;
        for (int idx_word = 0; idx_word < num_word_; idx_word++) {
          lkup_tbl_vec[idx_word] = *lkup_tbl_sel;
          lkup_tbl_sel += num_scbk_;
        }

        // Compute the gradient signal of the current sub-codebook
        caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_word_, len_word_,
            1, (Dtype)1., lkup_tbl_vec, bottom_data, (Dtype)1., scbk_sel);
        bottom_data += len_word_;
        scbk_sel += num_word_ * len_word_;
      }
    }
  }

  // If necessary, compute the gradient signal of the bias term
  if (bias_term_ && this->param_propagate_down_[2]) {
    const Dtype* top_diff = top[0]->cpu_diff();
    // Gradient with respect to bias
    caffe_cpu_gemv<Dtype>(CblasTrans, M_, N_, (Dtype)1., top_diff,
        bias_multiplier_.cpu_data(), (Dtype)1.,
        this->blobs_[2]->mutable_cpu_diff());
  }

  // Compute the gradient signal of the layer input
  if (propagate_down[0]) {
    // Compute the gradient signal for one instance at a time
    for (int m = 0; m < M_; m++) {
      // Compute the gradient signal of the look-up table
      caffe_set(lkup_tbl_.count(), (Dtype)0., lkup_tbl_.mutable_cpu_diff());
      const Dtype* top_diff = top[0]->cpu_diff() + m * N_;
      for (int idx_output = 0; idx_output < N_; idx_output++) {
        Dtype val = top_diff[idx_output];
        Dtype* lkup_tbl_sel = lkup_tbl_.mutable_cpu_diff();
        for (int idx_word = 0; idx_word < num_word_; idx_word++) {
          const vector<int>& scbk_idxs_sel = scbk_idxs_[idx_output][idx_word];
          for (std::size_t idx = 0; idx < scbk_idxs_sel.size(); idx++) {
            lkup_tbl_sel[scbk_idxs_sel[idx]] += val;
          }
        }
        lkup_tbl_sel += num_scbk_;
      }

      // Compute the gradient signal of the current instance's layer input
      Dtype* bottom_diff = bottom[0]->mutable_cpu_diff() + m * K_;
      const Dtype* scbk_sel = this->blobs_[0]->cpu_data();
      for (int idx_scbk = 0; idx_scbk < num_scbk_; idx_scbk++) {
        // Re-arrange the gradient signal of the look-up table
        const Dtype* lkup_tbl_sel = lkup_tbl_.cpu_diff() + idx_scbk;
        for (int idx_word = 0; idx_word < num_word_; idx_word++) {
          lkup_tbl_vec[idx_word] = *lkup_tbl_sel;
          lkup_tbl_sel += num_scbk_;
        }

        // Compute the gradient signal of the current instance's layer input
        caffe_cpu_gemv<Dtype>(CblasTrans, num_word_, len_word_, (Dtype)1.0,
            scbk_sel, lkup_tbl_vec, (Dtype)0., bottom_diff);
        bottom_diff += len_word_;
        scbk_sel += num_word_ * len_word_;
      }
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(QuanInnerProductLayer);
#endif

INSTANTIATE_CLASS(QuanInnerProductLayer);
REGISTER_LAYER_CLASS(QuanInnerProduct);

}  // namespace caffe
