struct CohModel <: TimeSeriesModel{2} end
function WhittleLikelihoodInference.add_sdf!(out, ::CohModel, ω)
    out[1] = 4
    out[2] = 3+4im
    out[3] = 9
end
@testset "coherancy" begin
    @test coherancy(CohModel(), 1.0) ≈ [1.0 (1/2-2im/3); (1/2+2im/3) 1.0]
    @test coherance(CohModel(), 1.0) ≈ [1.0 5/6; 5/6 1.0]
    @test groupdelay(CohModel(), 1.0) ≈ [0 -0.9272952180016122; 0.9272952180016122 0]
    @test_throws ErrorException coherancy(OU(1.0,1.0), 1.0)
    @test_throws ErrorException coherance(OU(1.0,1.0), 1.0)
    @test_throws ErrorException groupdelay(OU(1.0,1.0), 1.0)
end