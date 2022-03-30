function gradgen(f!,θ)
	G = zeros(length(θ))
	f!(nothing,G,nothing,θ)
	return G
end
@testset "standardwhittle" begin
    @test_throws ArgumentError WhittleLikelihood(OU,ones(10),-1)
    @test_throws ArgumentError WhittleLikelihood(OU,ones(10,2),1)
    for (model,par,ts) in zip((OU,TwoOU,CorrelatedOU,TwoCorrelatedOU),(ones(2),ones(4),[1,1,0.5],[1,1,0.5,1,1,0.5]),(ones(10),ones(10),ones(10,2),ones(10,2)))
        w = WhittleLikelihood(model,ts,1)
        @test_throws ArgumentError w(nothing,ones(length(par)+1),nothing,par)
        @test_throws ArgumentError w(nothing,nothing,ones(length(par),length(par)+1),par)
        @test w(par) isa Float64
        @test w(1,ones(length(par)),ones(length(par),length(par)),par) isa Float64
        @test w(nothing,ones(length(par)),ones(length(par),length(par)),par) == nothing
        @test grad(central_fdm(5, 1), w, par)[1] ≈ gradgen(w, par)
    end
end