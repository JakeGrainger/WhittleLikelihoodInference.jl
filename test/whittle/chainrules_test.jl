@testset "chainrules" begin
    models = (OU,OUUnknown{3},TwoOU,CorrelatedOU,CorrelatedOUUnknown{3},TwoCorrelatedOU)
    pars = (ones(2),ones(2),ones(4),[1,1,0.5],[1,1,0.5],[1,1,0.5,1,1,0.5])
    series = [fill(rand(10),3);fill(rand(10,2),3)]
    for (model,par,ts) in zip(models,pars,series)
        w = WhittleLikelihood(model,ts,1)
        dw = DebiasedWhittleLikelihood(model,ts,1)
        test_rrule(w ⊢ NoTangent(),par)
        test_rrule(dw ⊢ NoTangent(),par)
    end
end