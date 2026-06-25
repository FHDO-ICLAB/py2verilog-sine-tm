// =============================================================================
// binarizer.v  --  Thermometer Encoding for Tsetlin Machine Input
//
// Converts 13 MFCC coefficients into 104 binary features.
// Each coefficient is compared against 8 thresholds:
//   feature[c*8 + t] = (mfcc[c] > threshold[c][t])
//
// Thresholds are loaded from ROM: mem/binarizer_thresh.mem
//   Format: 13*8 = 104 entries, each 16-bit Q8.8 (same as MFCC)
//   Layout: coeff 0 thresholds 0..7, coeff 1 thresholds 0..7, ...
//
// Input:  13 MFCC values streamed in (in_idx 0..12)
// Output: 104-bit feature vector (pulsed valid once all 13 are processed)
// =============================================================================
`timescale 1ns / 1ps

module binarizer #(
    parameter N_COEFF      = 13,
    parameter N_THRESHOLDS = 8,
    parameter N_FEATURES   = 104   // N_COEFF * N_THRESHOLDS
)(
    input  wire         clk,
    input  wire         rst,
    input  wire         in_valid,
    input  wire [15:0]  in_data,   // Q8.8 signed MFCC
    input  wire [3:0]   in_idx,    // 0..12
    output wire         in_ready,

    input  wire         out_ready,
    output reg          out_valid,
    output reg [103:0]  out_features
);

    // -------------------------------------------------------------------------
    // Threshold ROM (104 entries x 16-bit)
    // -------------------------------------------------------------------------
    (* rom_style = "distributed" *)
    reg signed [15:0] thresh_rom [0:N_FEATURES-1];
    initial $readmemh("mem/binarizer_thresh.mem", thresh_rom);

    // -------------------------------------------------------------------------
    // MFCC buffer
    // -------------------------------------------------------------------------
    reg signed [15:0] mfcc_buf [0:N_COEFF-1];
    // -------------------------------------------------------------------------
    // Comparison state machine (sequential to save LUTs)
    // -------------------------------------------------------------------------
    localparam S_COLLECT = 2'd0;
    localparam S_CMP     = 2'd1;
    localparam S_OUT     = 2'd2;

    reg [1:0]   state    = S_COLLECT;
    reg [6:0]   feat_i   = 0;    // 0..103
    reg [103:0] feat_reg = 0;
    reg [103:0] feat_next;

    assign in_ready = (state == S_COLLECT);

    always @(posedge clk) begin
        if (rst) begin
            state     <= S_COLLECT;
            out_valid <= 0;
        end else begin
            case (state)

            S_COLLECT: begin
                out_valid <= 0;

                if (in_valid) begin
                    mfcc_buf[in_idx] <= in_data;

                    if (in_idx == N_COEFF - 1) begin
                        feat_i   <= 0;
                        feat_reg <= 0;
                        state    <= S_CMP;
                    end
                end
            end

            S_CMP: begin
                feat_next = feat_reg;
                feat_next[feat_i] = ($signed(mfcc_buf[feat_i[6:3]]) >
                                     $signed(thresh_rom[feat_i]));

                feat_reg <= feat_next;

                if (feat_i == N_FEATURES - 1) begin
                    out_features <= feat_next;
                    state        <= S_OUT;
                end else begin
                    feat_i <= feat_i + 1;
                end
            end

            S_OUT: begin
                if (!out_valid) begin
                    out_valid <= 1;
                end else if (out_ready) begin
                    out_valid <= 0;
                    feat_i   <= 0;
                    feat_reg <= 0;
                    state    <= S_COLLECT;
                end
            end

            endcase
        end
    end

endmodule
