// =============================================================================
// tb_tm_pipeline.v
//
// Testbench for the minimal MFCC -> Binarizer -> TM classifier pipeline.
//
// Before running simulation:
//   1. Generate mem/mfcc_frames.mem:
//      python3 wav_to_mfcc_mem.py test.wav
//
//   2. Put these files in the simulation directory:
//      mem/mfcc_frames.mem
//      mem/binarizer_thresh.mem
//      mem/tm_states.mem
//
//   3. Compile:
//      iverilog -g2012 -o sim \
//        tb_tm_pipeline.v tm_pipeline.v mfcc_frame_source.v \
//        binarizer.v tm_classifier.v
//
//   4. Run:
//      vvp sim
// =============================================================================

`timescale 1ns / 1ps

module tb_tm_pipeline;

    parameter N_CLASSES  = 5;
    parameter MAX_FRAMES = 95;    // Adjust to the number of frames in mem/mfcc_frames.mem
    parameter TIMEOUT_CYCLES = 5000000;

    reg clk = 0;
    reg rst = 1;
    reg start = 0;

    wire out_valid;
    wire [2:0] out_class;
    wire frame_done;
    wire all_done;
    wire [31:0] frame_number;

    integer votes [0:N_CLASSES-1];
    integer i;
    integer best_class;
    integer best_votes;
    integer total_predictions;
    integer cycles;

    tm_pipeline #(
        .N_CLASSES(N_CLASSES),
        .MAX_FRAMES(MAX_FRAMES)
    ) dut (
        .clk(clk),
        .rst(rst),
        .start(start),

        .out_valid(out_valid),
        .out_class(out_class),

        .frame_done(frame_done),
        .all_done(all_done),
        .frame_number(frame_number)
    );

    always #5 clk = ~clk; // 100 MHz

    initial begin
        for (i = 0; i < N_CLASSES; i = i + 1)
            votes[i] = 0;

        total_predictions = 0;

        #100;
        rst = 0;

        @(posedge clk);
        start = 1;
        @(posedge clk);
        start = 0;

        for (cycles = 0;
             cycles < TIMEOUT_CYCLES && total_predictions < MAX_FRAMES;
             cycles = cycles + 1) begin
            @(posedge clk);

            if (out_valid) begin
                votes[out_class] = votes[out_class] + 1;
                total_predictions = total_predictions + 1;
                $display("Frame prediction %0d -> class %0d",
                         total_predictions, out_class);
            end
        end

        if (total_predictions != MAX_FRAMES) begin
            $display("");
            $display("WARNING: expected %0d predictions, got %0d",
                     MAX_FRAMES, total_predictions);
        end

        best_class = 0;
        best_votes = votes[0];

        for (i = 1; i < N_CLASSES; i = i + 1) begin
            if (votes[i] > best_votes) begin
                best_votes = votes[i];
                best_class = i;
            end
        end

        $display("");
        $display("Vote summary:");
        for (i = 0; i < N_CLASSES; i = i + 1)
            $display("  class %0d: %0d", i, votes[i]);

        $display("");
        $display("Final predicted class = %0d", best_class);

        $finish;
    end

endmodule
