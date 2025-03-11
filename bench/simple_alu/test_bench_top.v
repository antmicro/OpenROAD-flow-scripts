`timescale 1 ns / 1 ns
module tb;
    reg clk = 0;
    reg[7:0] a = 0;
    reg[7:0] b = 0;
    reg[1:0] op;
    wire[7:0] out;
    simple_alu uut(.*);

    always #1 clk = ~clk;
    initial begin
        #50
        a = 42;
        b = 99;
        op = 0 /*ADD*/;
        #50
        if (out != 42 + 99) $stop;
        $display("a=%0d, b=%0d, op=%0d, out=%0d", a, b, op, out);

        a = 99;
        b = 42;
        op = 1 /*SUB*/;
        #50
        if (out != 99 - 42) $stop;
        $display("a=%0d, b=%0d, op=%0d, out=%0d", a, b, op, out);

        a = 1;
        b = 0;
        op = 2 /*AND*/;
        #50
        if (out != 0) $stop;
        $display("a=%0d, b=%0d, op=%0d, out=%0d", a, b, op, out);

        a = 1;
        b = 0;
        op = 3 /*OR*/;
        #50
        if (out != 1) $stop;
        $display("a=%0d, b=%0d, op=%0d, out=%0d", a, b, op, out);
    end

    initial #10000 $finish;
endmodule
