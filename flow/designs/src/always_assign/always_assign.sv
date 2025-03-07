module always_assign(
    input clk,
    input a,
    output out);

    always @(posedge clk) out = a;
endmodule
