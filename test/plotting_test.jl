using Plots
@testset "plotting" begin
    @testset "plotsdf" begin
        @test_throws ArgumentError plotsdf(1.0)
        @test_throws ArgumentError plotsdf(OU,1:2)
        @test_throws ArgumentError plotsdf(1.0,1:2)
        @test_throws ArgumentError plotsdf(OU(1.0,1.0),1)
        @test_throws ArgumentError plotsdf(OU(1.0,1.0),1:2,1)
    end
    @testset "plotasdf" begin
        @test_throws ArgumentError plotasdf(1.0)
        @test_throws ArgumentError plotasdf(OU,1:2,1)
        @test_throws ArgumentError plotasdf(OU(1.0,1.0),1:2)
        @test_throws ArgumentError plotasdf(OU(1.0,1.0),1:2,-1)
        @test_throws ArgumentError plotasdf(OU(1.0,1.0),1:2,-1,1)
    end
    @testset "plotacv" begin
        @test_throws ArgumentError plotacv(1.0)
        @test_throws ArgumentError plotacv(OU,1:2)
        @test_throws ArgumentError plotacv(OUUnknown(1.0,1.0),1:2)
        @test_throws ArgumentError plotacv(OU,10,1)
        @test_throws ArgumentError plotacv(OU(1.0,1.0),1:2,1)
        @test_throws ArgumentError plotacv(OU(1.0,1.0),10,-1)
        @test_throws ArgumentError plotacv(OU(1.0,1.0),-10,1)
        @test_throws ArgumentError plotacv(OU(1.0,1.0),10,1im)
        @test_throws ArgumentError plotacv(OU(1.0,1.0),10,1,1)
    end
    @testset "plotei" begin
        @test_throws ArgumentError plotei(1.0)
        @test_throws ArgumentError plotei(OU,10,1)
        @test_throws ArgumentError plotei(OU(1.0,1.0),10)
        @test_throws ArgumentError plotei(OU(1.0,1.0),1:2,1)
        @test_throws ArgumentError plotei(OU(1.0,1.0),10,-1)
        @test_throws ArgumentError plotei(OU(1.0,1.0),-10,1)
        @test_throws ArgumentError plotei(OU(1.0,1.0),10,1im)
        @test_throws ArgumentError plotei(OU(1.0,1.0),10,1,1)
    end
end