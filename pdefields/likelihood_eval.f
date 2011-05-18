! Subroutine lkinit initializes the log-likelihood for gmrfmetro. It also uses the template parameter LIKELIHOOD_CODE.
      subroutine lkinit (nx,x,lk, M, lv, nv)
!f2py intent(hide) nx, nv
!f2py intent(out) lk
      implicit none
      double precision x(nx), lk(nx), M(nx), lkp, lv(nx,nv), xp
      integer nx, i, nv

!       x is the current state. The log-likelihood will be returned for it.
!       M is the mean vector.
!       lv are the other vertex-specific variables used in computing the likelihood.
!       They should be stored as an (nx, nv) array, where nv is the
!       number of variables.
!
!       The return value will be a vector of log-likelihoods.

      do i=1,nx
          
!         This is a template parameter that gets substituted for at runtime. It should contain Fortran code that computes the log-likelihood, in terms of {X}, i, lv and lkp. {X} will be replaced with (xp+M(i)), that is the actual proposed value.
          xp = x(i) 
          
          {LIKELIHOOD_CODE}

          lk(i) = lkp
          
      end do
      
      return
      end