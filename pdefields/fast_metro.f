! Subroutine gmrfmetro is a template for a compiled Metropolis sweep of a Gaussian Markov random field. The template parameter LPF in braces should be replaced with a code snippet that evaluates the log-likelihood of a proposal, and the resulting code can be compiled either ahead of time or on the fly.
! 
! The subroutine takes as arguments the prior precision matrix in compressed sparse row or column format; the current value; the current log-probability; any vertex-specific data needed to evaluate likelihoods; and random variates for constructing the proposals and for deciding acceptance.

      subroutine gmrfmetro (ind,dat,ptr,nnz,nx,x,lp,
     & diag, M, acc, norms, lv, nv)
!f2py intent(hide) nnz, nx, nv
!f2py intent(inplace) lp, x
      implicit none
      double precision x(nx), dat(nnz), lp(nx), M(nx)
      double precision diag(nx), acc(nx), norms(nx), mc(nx)
      double precision xp, lpp, lv(nx,nv)
      integer ind(nnz), ptr(nx+1), nx, nnz, i, j, nv
!       x and lp are the current state and log-probability values.
!       They will both be overwritten IN-PLACE.
!
!       ind, dat, ptr are the sparse precision matrix.
!       diag is its diagonal.
!       norms is a vector of N(0,1) proposal deviates.
!       acc is a vector of Uniform(0,1) acceptance deviates.
!       lv are the other vertex-specific variables used in computing the likelihood.
!       They should be stored as an (nx, nv) array, where nv is the
!       number of variables.
      do i=1,nx
!         mc is the conditional mean of x(i),
!         (Q(i,i) x(i) - Q(i,:) x) / Q(i,i)
          mc(i) = diag(i)*x(i)
          do j=ptr(i)+1,ptr(i+1)
              mc(i) = mc(i) - dat(j)*x(ind(j)+1)
          end do
          mc(i) = mc(i)/diag(i)
          
!         Propose from the conditional prior.
          xp = norms(i)/dsqrt(diag(i))+mc(i)
          
!         This is a template parameter that gets substituted for at runtime. It should contain Fortran code that computes the likelihood, in terms of {X}, i, lv and lp. {X} will be replaced with (xp+M(i)), that is the actual proposed value.
          {LPF}
          
!         Accept or reject.
          if ((lpp-lp(i)).GT.dlog(acc(i))) then
              lp(i)=lpp
              x(i)=xp
          end if
          
      end do
      
      return
      end
  