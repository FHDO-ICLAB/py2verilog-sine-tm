// =============================================================================
// mfcc_frame_source.v
//
// Reads precomputed MFCC coefficients from mem/mfcc_frames.mem and streams them
// into the binarizer.
//
// Memory layout expected:
//   frame0 coeff0
//   frame0 coeff1
//   ...
//   frame0 coeff12
//   frame1 coeff0
//   ...
//
// Each value is signed 16-bit Q8.8 hexadecimal.
// =============================================================================

`timescale 1ns / 1ps

module mfcc_frame_source #(
    parameter N_COEFF    = 13,
    parameter MAX_FRAMES = 1024,
    parameter MEM_DEPTH  = N_COEFF * MAX_FRAMES
)(
    input  wire        clk,
    input  wire        rst,
    input  wire        start,
    input  wire        out_ready,

    output reg         out_valid,
    output reg [15:0]  out_data,
    output reg [3:0]   out_idx,

    output reg         frame_done,
    output reg         all_done,
    output reg [31:0]  frame_number
);

    reg [15:0] mfcc_mem [0:MEM_DEPTH-1];

    initial begin
        $readmemh("mem/mfcc_frames.mem", mfcc_mem);
    end

    localparam S_IDLE = 2'd0;
    localparam S_SEND = 2'd1;
    localparam S_DONE = 2'd2;

    reg [1:0] state = S_IDLE;
    reg [31:0] mem_ptr = 0;
    reg [3:0] coeff_idx = 0;

    always @(posedge clk) begin
        out_valid  <= 0;
        frame_done <= 0;
        all_done   <= 0;

        if (rst) begin
            state        <= S_IDLE;
            mem_ptr      <= 0;
            coeff_idx    <= 0;
            frame_number <= 0;
        end else begin
            case (state)

            S_IDLE: begin
                if (start) begin
                    state        <= S_SEND;
                    mem_ptr      <= 0;
                    coeff_idx    <= 0;
                    frame_number <= 0;
                    out_data     <= mfcc_mem[0];
                    out_idx      <= 0;
                    out_valid    <= 1;
                end
            end

            S_SEND: begin
                out_valid <= 1;

                if (out_ready) begin
                    if (coeff_idx == N_COEFF - 1) begin
                        frame_done  <= 1;
                        frame_number <= frame_number + 1;
                    end

                    if (mem_ptr == MEM_DEPTH - 1) begin
                        out_valid <= 0;
                        state     <= S_DONE;
                    end else begin
                        mem_ptr <= mem_ptr + 1;
                        out_data <= mfcc_mem[mem_ptr + 1];

                        if (coeff_idx == N_COEFF - 1) begin
                            coeff_idx <= 0;
                            out_idx   <= 0;
                        end else begin
                            coeff_idx <= coeff_idx + 1;
                            out_idx   <= coeff_idx + 1;
                        end
                    end
                end
            end

            S_DONE: begin
                all_done <= 1;
            end

            endcase
        end
    end

endmodule
