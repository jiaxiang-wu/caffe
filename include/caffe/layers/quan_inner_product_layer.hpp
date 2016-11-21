#ifndef CAFFE_QUAN_INNER_PRODUCT_LAYER_HPP_
#define CAFFE_QUAN_INNER_PRODUCT_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

/**
 * @brief The parameter-quantized version of InnerProduct layer, also known as
 *        a "fully-connected" layer, computes an inner product with a set of
 *        learned sub-codebooks and quantization indicators, and (optionally)
 *        adds biases.
 *
 * TODO(dox): thorough documentation for Forward, Backward, and proto params.
 */
template <typename Dtype>
class QuanInnerProductLayer : public Layer<Dtype> {
 public:
  explicit QuanInnerProductLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "QuanInnerProduct"; }
  virtual inline int ExactNumBottomBlobs() const { return 1; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  int len_word_;  // length of each codeword
  int num_word_;  // number of codewords in each sub-codebook
  int num_scbk_;  // number of subspaces / sub-codebooks
  int M_;  // number of samples within the mini-batch
  int K_;  // number of input dimensions
  int N_;  // number of output dimensions
  bool bias_term_;  // whether the bias term exists
  Blob<Dtype> bias_multiplier_;  // use multiplication to add biases

  Blob<Dtype> lkup_tbl_;  // look-up table of pre-computed inner products
};

}  // namespace caffe

#endif  // CAFFE_QUAN_INNER_PRODUCT_LAYER_HPP_
