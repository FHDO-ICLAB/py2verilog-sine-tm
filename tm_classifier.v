// =============================================================================
// tm_classifier.v  --  Tsetlin Machine Classifier (Inference Only)
//
// Architecture:
//   - N_CLASSES  classes (default 5)
//   - N_CLAUSES  clauses per class (default 20)
//   - N_FEATURES binary features (default 104)
//   - Each clause has 2*N_FEATURES automata (one per literal)
//   - Automaton action = include literal  if state >= N_STATES/2
//   - Clause output = AND of all included literals
//   - Class score  = sum of first-half clause outputs
//                  - sum of second-half clause outputs
//   - Predicted class = argmax(score)
//
// Automaton states stored in BRAM:
//   Address layout: class * N_CLAUSES * 2*N_FEATURES
//                   + clause * 2*N_FEATURES
//                   + literal_idx
//   Width: N_STATE_BITS bits (default 4 -> 16 states)
//
// States are loaded from  mem/tm_states.mem  (generated after training)
//
// Processing: sequential over all clauses and literals.
// Latency: N_CLASSES * N_CLAUSES * 2 * N_FEATURES clock cycles per frame
//   = 5 * 20 * 208 = 20800 cycles  (0.2 ms @ 100 MHz) -- well within budget
// =============================================================================
`timescale 1ns / 1ps

module tm_classifier #(
    parameter N_CLASSES     = 5,
    parameter N_CLAUSES     = 20,
    parameter N_FEATURES    = 104,
    parameter N_LITERALS    = 208,   // 2 * N_FEATURES
    parameter N_STATES      = 16,
    parameter N_STATE_BITS  = 4,
    parameter T             = 15     // voting threshold (not used in inference)
)(
    input  wire          clk,
    input  wire          rst,
    input  wire          in_valid,
    input  wire [103:0]  in_features,   // 104-bit binary feature vector
    output wire          in_ready,

    output reg           out_valid,
    output reg  [2:0]    out_class      // 0 .. N_CLASSES-1
);

    // -------------------------------------------------------------------------
    // Automaton state BRAM
    // Depth = N_CLASSES * N_CLAUSES * N_LITERALS = 5 * 20 * 208 = 20800
    // Width = N_STATE_BITS = 4 bits
    // -------------------------------------------------------------------------
    localparam MEM_DEPTH = N_CLASSES * N_CLAUSES * N_LITERALS;

    (* ram_style = "block" *)
    reg [N_STATE_BITS-1:0] state_mem [0:MEM_DEPTH-1];
    initial $readmemh("mem/tm_states.mem", state_mem);

    // -------------------------------------------------------------------------
    // Literal evaluation helper
    // literal index:  0..N_FEATURES-1     -> positive literal  feat[i]
    //                 N_FEATURES..2*N_FEATURES-1 -> negative literal  ~feat[i]
    // -------------------------------------------------------------------------
    function literal_val;
        input [7:0]   lit_idx;
        input [103:0] features;
        begin
            if (lit_idx < N_FEATURES)
                literal_val = features[lit_idx];
            else
                literal_val = ~features[lit_idx - N_FEATURES];
        end
    endfunction

    // -------------------------------------------------------------------------
    // State machine
    // -------------------------------------------------------------------------
    localparam S_IDLE   = 3'd0;
    localparam S_CLAUSE = 3'd1;   // evaluate one clause (iterate literals)
    localparam S_VOTE   = 3'd2;   // accumulate class vote
    localparam S_ARGMAX = 3'd3;   // find winning class
    localparam S_OUT    = 3'd4;

    reg [2:0]  state      = S_IDLE;

    // Loop indices
    reg [2:0]  cls        = 0;   // 0..N_CLASSES-1
    reg [4:0]  clause     = 0;   // 0..N_CLAUSES-1
    reg [7:0]  lit        = 0;   // 0..N_LITERALS-1

    // Per-class vote accumulators (signed, range -N_CLAUSES..N_CLAUSES)
    reg signed [5:0]  votes [0:N_CLASSES-1];
    reg               clause_out = 0;   // current clause result

    // BRAM read
    reg [14:0] mem_addr;
    reg [N_STATE_BITS-1:0] st_val;
    reg        include_lit;         // state >= N_STATES/2
    reg [103:0] feat_lat;

    // Argmax
    reg [2:0]  best_cls   = 0;
    reg signed [5:0] best_score = -32;

    integer c;

    assign in_ready = (state == S_IDLE);

    always @(posedge clk) begin
        out_valid <= 0;

        if (rst) begin
            state <= S_IDLE;
            for (c = 0; c < N_CLASSES; c = c+1)
                votes[c] <= 0;
        end else begin
            case (state)

            // -----------------------------------------------------------------
            S_IDLE: begin
                if (in_valid) begin
                    feat_lat <= in_features;
                    cls      <= 0;
                    clause   <= 0;
                    lit      <= 0;
                    clause_out <= 1;   // AND identity
                    for (c = 0; c < N_CLASSES; c = c+1)
                        votes[c] <= 0;
                    state <= S_CLAUSE;
                end
            end

            // -----------------------------------------------------------------
            // Evaluate one literal per cycle
            // -----------------------------------------------------------------
            S_CLAUSE: begin
                mem_addr  = cls * (N_CLAUSES * N_LITERALS)
                          + clause * N_LITERALS
                          + lit;
                st_val    = state_mem[mem_addr];
                include_lit = (st_val >= N_STATES / 2);

                // If literal is included and its value is 0, clause = 0
                if (include_lit && !literal_val(lit[7:0], feat_lat))
                    clause_out <= 0;

                if (lit == N_LITERALS - 1) begin
                    lit   <= 0;
                    state <= S_VOTE;
                end else begin
                    lit <= lit + 1;
                end
            end

            // -----------------------------------------------------------------
            S_VOTE: begin
                // tmu.TMClassifier uses first half positive, second half negative.
                if (clause < (N_CLAUSES / 2))
                    votes[cls] <= votes[cls] + clause_out;
                else
                    votes[cls] <= votes[cls] - clause_out;

                clause_out <= 1;   // reset for next clause

                if (clause == N_CLAUSES - 1) begin
                    clause <= 0;
                    if (cls == N_CLASSES - 1) begin
                        // All classes done -> argmax
                        best_cls   <= 0;
                        best_score <= votes[0];
                        cls        <= 1;
                        state      <= S_ARGMAX;
                    end else begin
                        cls    <= cls + 1;
                        state  <= S_CLAUSE;
                    end
                end else begin
                    clause <= clause + 1;
                    state  <= S_CLAUSE;
                end
            end

            // -----------------------------------------------------------------
            S_ARGMAX: begin
                if (cls < N_CLASSES) begin
                    if ($signed(votes[cls]) > best_score) begin
                        best_score <= votes[cls];
                        best_cls   <= cls[2:0];
                    end
                    cls <= cls + 1;
                end else begin
                    state <= S_OUT;
                end
            end

            // -----------------------------------------------------------------
            S_OUT: begin
                out_class <= best_cls;
                out_valid <= 1;
                state     <= S_IDLE;
            end

            endcase
        end
    end

endmodule
