`timescale 1 ns / 1 ns
module tb;
    reg clk = 0;
    reg  a = 0;
    wire out;

    always_assign uut(.clk(clk), .a(a), .out(out));

    always #1 clk = ~clk;
    initial begin
        #1000;
        $monitor("%d %d", a, out);
        if (out != 0) begin
            $stop;
        end

        a = 1;
        #1000;

        if (out != 1) begin
            $stop;
        end
    end

    initial #10000 $finish;

endmodule

