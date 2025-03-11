typedef enum bit[1:0] {
    ADD, SUB, AND, OR
} Op;

module simple_alu (
    input clk,
    input[7:0] a,
    input[7:0] b,
    input Op op,
    output reg[7:0] out);

    always @(posedge clk)
        case(op)
            ADD: out <= a + b;
            SUB: out <= a - b;
            AND: out <= a & b;
            OR: out <= a | b;
        endcase
endmodule
