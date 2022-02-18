const OUDouble = OU+OU
@testset "typestructure" begin
    @testset "AdditiveTimeSeriesModel" begin
        @test AdditiveTimeSeriesModel(OU(1,1),OU(1,1)) == AdditiveTimeSeriesModel{OU,OU,1,Float64}(ones(4))
        @test_throws MethodError AdditiveTimeSeriesModel(OU(1,1),CorrelatedOU(1,1,1/2))
        @test OUDouble == AdditiveTimeSeriesModel{OU,OU,1,Float64}
        @test_throws MethodError OU+CorrelatedOU
        @test OU(1,1)+OU(1,1) == AdditiveTimeSeriesModel{OU,OU,1,Float64}(ones(4))
    end
    @testset "ndims" begin
        @test ndims(OU) == 1
        @test ndims(Matern{10,55}) == 10
        @test ndims(CorrelatedOU(1,1,1/2)) == 2
    end
    @testset "nlowertriangle_dimension" begin
        @test nlowertriangle_dimension(OU) == 1
        @test nlowertriangle_dimension(Matern{10,55}) == 55
        @test nlowertriangle_dimension(CorrelatedOU(1,1,1/2)) == 3
    end
    @testset "npars" begin
        @test npars(OU) == 2
        @test npars(CorrelatedOU(1,1,1/2)) == 3
        @test npars(OUDouble) == 4
    end
    @testset "nlowertriangle_parameter" begin
        @test nlowertriangle_parameter(OU) == 3
        @test nlowertriangle_parameter(CorrelatedOU(1,1,1/2)) == 6
        @test nlowertriangle_parameter(OUDouble) == 10
    end
    @testset "indexLT" begin
        @test indexLT(2,3,3) == 5
        @test indexLT(4,2,5) == 8
    end
    @testset "nalias" begin
        @test nalias(OU(1,1)) isa Int
    end
    @testset "minbins" begin
        @test minbins(OU) isa Int
    end
end
