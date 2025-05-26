#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <windows.h>
#include <cmath>

class Transformer {
    // Model constants
    int dim{};
    int hidden_dim{};
    int n_layers{};
    int n_heads{};
    int n_kv_heads{};
    int vocab_size{};
    int seq_len{};
    int head_size{};

    // Learned parameters
    std::vector<std::vector<double>> token_embedding;
    std::vector<std::vector<double>> rms_att_weight;
    std::vector<std::vector<double>> rms_ffn_weight;
    std::vector<double> rms_final_weight;
    std::vector<std::vector<double>> freq_cis_real;
    std::vector<std::vector<double>> freq_cis_imag;
    std::vector<std::vector<std::vector<double>>> wq;
    std::vector<std::vector<std::vector<double>>> wv;
    std::vector<std::vector<std::vector<double>>> wk;
    std::vector<std::vector<std::vector<double>>> wo;
    std::vector<std::vector<std::vector<double>>> w1;
    std::vector<std::vector<std::vector<double>>> w2;
    std::vector<std::vector<std::vector<double>>> w3;

    std::vector<std::string> vocab;

    // Key/value caches (3D: layer, token, vector)
    std::vector<std::vector<std::vector<double>>> key_cache;
    std::vector<std::vector<std::vector<double>>> value_cache;

    // Intermediate vectors
    std::vector<double> x;
    std::vector<double> xb;
    std::vector<double> xb2;
    std::vector<double> hb;
    std::vector<double> hb2;
    std::vector<double> q;
    std::vector<double> k;
    std::vector<double> v;
    std::vector<double> attention;
    std::vector<double> logits;

    // Readers
    std::vector< std::vector<std::vector<double>> > Read3D(int dim1, int dim2, int dim3, std::ifstream& stream) {
        std::vector< std::vector<std::vector<double>> > output(dim1);
        for (int i = 0; i < dim1; i++) {
            output[i] = Read2D(dim2, dim3, stream);
        }
        return output;
    }

    std::vector<std::vector<double>> Read2D(int dim1, int dim2, std::ifstream& stream) {
        std::vector<std::vector<double>> output(dim1);
        for (int i = 0; i < dim1; i++) {
            output[i] = Read1D(dim2, stream);
        }
        return output;
    }

    std::vector<double> Read1D(int dim, std::ifstream& stream) {
        std::vector<double> output(dim);
        for (int i = 0; i < dim; i++) {
            float val{};
            stream.read(reinterpret_cast<char*>(&val), sizeof(float));
            output[i] = static_cast<double>(val);
        }
        return output;
    }

    void rmsnorm(std::vector<double> &output, std::vector<double> &x, std::vector<double> &weight) {
        double ss = 0.0;
        for (int i = 0; i < x.size(); i++) {
            double sqr = x[i] * x[i];
            ss += sqr;
        }

        ss /= static_cast<double>(x.size());
        ss += 1e-5;
        
        ss = 1.0 / sqrt(ss);

        for (int i = 0; i < x.size(); i++) {
            output[i] = weight[i] * (ss * x[i]);
        }
    }

    void softmax(std::vector<double> &output, std::vector<double> &input, int pos) {
        double max_val = input[0];
        for (int i = 1; i <= pos; i++) {
            if (input[i] > max_val) {
                max_val = input[i];
            }
        }

        double sum = 0.0;
        for (int i = 0; i <= pos; i++) {
            output[i] = std::exp(input[i] - max_val);
            sum += output[i];
        }

        for (int i = 0; i <= pos; i++) {
            output[i] /= sum;
        }
    }

    void matmul(std::vector<double> &output, std::vector<double> &x, std::vector<std::vector<double>> &w) {
        for (int i = 0; i < output.size(); i++) {

            double val = 0.0;
            for (int j = 0; j < x.size(); j++) {
                val += w[i][j] * x[j];
            }

            output[i] = val;
        }
    }

    void accum(std::vector<double> &lhs, std::vector<double> &rhs) {
        for (int i = 0; i < lhs.size(); i++) {
            lhs[i] += rhs[i];
        }
    }

    void copy(std::vector<double> &dst, const std::vector<double> &src) {
        std::copy(src.begin(), src.end(), dst.begin());
    }

    void transformer(int next_token, int pos) {
        copy(x, token_embedding[next_token]);

        for (int layer = 0; layer < n_layers; layer++) {

            rmsnorm(xb, x, rms_att_weight[layer]);

            matmul(q, xb, wq[layer]);
            matmul(k, xb, wk[layer]);
            matmul(v, xb, wv[layer]);

            for (int head = 0; head < n_heads; head++) {

                for (int i = 0; i < head_size; i += 2) {

                    double fcr = freq_cis_real[pos][i / 2];
                    double fci = freq_cis_imag[pos][i / 2];

                    double q0 = q[head * head_size + i];
                    double q1 = q[head * head_size + i + 1];
                    double k0 = k[head * head_size + i];
                    double k1 = k[head * head_size + i + 1];

                    q[head * head_size + i] = q0 * fcr - q1 * fci;
                    q[head * head_size + i + 1] = q0 * fci + q1 * fcr;
                    k[head * head_size + i] = k0 * fcr - k1 * fci;
                    k[head * head_size + i + 1] = k0 * fci + k1 * fcr;
                }
            }

            for (int i = 0; i < dim; i++) {
                key_cache[layer][pos][i] = k[i];
                value_cache[layer][pos][i] = v[i];
            }

            for (int head = 0; head < n_heads; ++head) {

                for (int token_pos = 0; token_pos <= pos; token_pos++) {
                    double score = 0.0;
                    for (int i = 0; i < head_size; i++) {
                        score += q[head * head_size + i] * key_cache[layer][token_pos][head * head_size + i];
                    }
                    score /= sqrt(head_size);
                    attention[token_pos] = score;
                }

                softmax(attention, attention, pos);

                for (int i = 0; i < head_size; ++i) {
                    double val = 0.0;
                    for (int token = 0; token <= pos; ++token) {
                        val += attention[token] * value_cache[layer][token][head * head_size + i];
                    }
                    xb[head * head_size + i] = val;
                }
            }

            matmul(xb2, xb, wo[layer]);
            accum(x, xb2);
            rmsnorm(xb, x, rms_ffn_weight[layer]);
            matmul(hb, xb, w1[layer]);
            matmul(hb2, xb, w3[layer]);

            for (int i = 0; i < hidden_dim; i++) {
                double sigmoid = 1.0 / (1.0 + std::exp(-hb[i]));
                hb[i] = hb[i] * sigmoid;
            }

            for (int i = 0; i < hidden_dim; i++) {
                hb[i] = hb[i] * hb2[i];
            }

            matmul(xb, hb, w2[layer]);
            accum(x, xb);
        }

        rmsnorm(x, x, rms_final_weight);
        matmul(logits, x, token_embedding);
    }

    int argmax(std::vector<double> &v) {
        int max_i = 0;
        double max_p = v[0];
        for (int i = 1; i < v.size(); i++) {
            if (v[i] > max_p) {
                max_i = i;
                max_p = v[i];
            }
        }
        return max_i;
    }
    
    public:
        Transformer(){
            // Reading from model file
            std::ifstream modelreader("model.bin", std::ios::binary);

            modelreader.read(reinterpret_cast<char*>(&dim), sizeof(dim));
            modelreader.read(reinterpret_cast<char*>(&hidden_dim), sizeof(hidden_dim));
            modelreader.read(reinterpret_cast<char*>(&n_layers), sizeof(n_layers));
            modelreader.read(reinterpret_cast<char*>(&n_heads), sizeof(n_heads));
            modelreader.read(reinterpret_cast<char*>(&n_kv_heads), sizeof(n_kv_heads));
            modelreader.read(reinterpret_cast<char*>(&vocab_size), sizeof(vocab_size));
            modelreader.read(reinterpret_cast<char*>(&seq_len), sizeof(seq_len));

            token_embedding = Read2D(vocab_size, dim, modelreader);
            rms_att_weight = Read2D(n_layers, dim, modelreader);
            wq = Read3D(n_layers, dim, dim, modelreader);
            wk = Read3D(n_layers, dim, dim, modelreader);
            wv = Read3D(n_layers, dim, dim, modelreader);
            wo = Read3D(n_layers, dim, dim, modelreader);
            rms_ffn_weight = Read2D(n_layers, dim, modelreader);
            w1 = Read3D(n_layers, hidden_dim, dim, modelreader);
            w2 = Read3D(n_layers, dim, hidden_dim, modelreader);
            w3 = Read3D(n_layers, hidden_dim, dim, modelreader);
            rms_final_weight = Read1D(dim, modelreader);
            head_size = dim / n_heads;
            freq_cis_real = Read2D(seq_len, head_size / 2, modelreader);
            freq_cis_imag = Read2D(seq_len, head_size / 2, modelreader);

            // Reading from token file
            std::ifstream tokenreader("tokenizer.bin", std::ios::binary);
            vocab = std::vector<std::string>(vocab_size);

            for (int i = 0; i < vocab_size; i++){
                int len{};
                tokenreader.read(reinterpret_cast<char*>(&len), sizeof(int));
                std::vector<char> bytes(len);
                tokenreader.read(bytes.data(), len);
                std::string chars(bytes.begin(), bytes.end());
                vocab[i] = chars;
            }

            x = std::vector<double>(dim);
            xb = std::vector<double>(dim);
            xb2 = std::vector<double>(dim);
            hb = std::vector<double>(hidden_dim);
            hb2 = std::vector<double>(hidden_dim);
            q = std::vector<double>(dim);
            k = std::vector<double>(dim);
            v = std::vector<double>(dim);
            attention = std::vector<double>(seq_len);
            logits = std::vector<double>(vocab_size);

            key_cache = std::vector<std::vector<std::vector<double>>>(
                n_layers, std::vector<std::vector<double>>(seq_len, std::vector<double>(dim)));

            value_cache = std::vector<std::vector<std::vector<double>>>(
                n_layers, std::vector<std::vector<double>>(seq_len, std::vector<double>(dim)));
        }

        void tellStory() {
            int token = 1;
            int pos = 0;

            while (pos < seq_len) {
                transformer(token, pos);
                int next = argmax(logits);

                if (next == 1) {
                    break;
                }

                std::cout << vocab[next];

                token = next;
                ++pos;
            }
        }
};

int main()
{
    SetConsoleOutputCP(CP_UTF8);
    Transformer *tf = new Transformer();
    tf->tellStory();
    return 0;
}

